import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # K block pointer: will be advanced each iteration, shape (K_TILE_SIZE, D)
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # V block pointer: will be advanced each iteration, shape (K_TILE_SIZE, D)
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # O block pointer
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L block pointer
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q data
    Q_tile = tl.load(Q_block_ptr)

    # set O_i, m_i, l_i
    O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)

    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    # T_k loop
    for t_k in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr)
        V_tile = tl.load(V_block_ptr)

        # compute S_tile of pre-softmax attention
        S_tile = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # apply causal mask if needed
        if is_causal:
            k_offsets = t_k * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            S_tile = tl.where(causal_mask, S_tile, float("-1e6"))
        # update m_ij
        m_ij = tl.maximum(m_i, tl.max(S_tile, axis=-1))
        # compute P_tile
        P_tile = tl.exp(S_tile - m_ij[:,None])
        # update l_i
        l_i = tl.exp(m_i - m_ij) * l_i + tl.sum(P_tile, axis=-1)
        # update O_tile
        O_tile = tl.exp(m_i - m_ij)[:,None] * O_tile + tl.dot(P_tile.to(V_tile.dtype), V_tile)   # cast P_tile to the dtype of V_tile before they are multiplied
        # update m_i
        m_i = m_ij
        # advance K_block_ptr and V_block_ptr
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_tile = O_tile / l_i[:,None]
    L_tile = m_i + tl.log(l_i)
    
    # store th compute results
    tl.store(O_block_ptr, O_tile.to(O_block_ptr.type.element_ty))    # cast O_tile to the appropriate dtype before store to global memory
    tl.store(L_block_ptr, L_tile.to(L_block_ptr.type.element_ty))


class FlashAttentionTriton(torch.autograd.Function):
    """Triton-accelerated FlashAttention-2 forward pass."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Args:
            Q: (batch, N_q, d) on CUDA
            K: (batch, N_k, d) on CUDA
            V: (batch, N_k, d) on CUDA
            is_causal: bool
        Returns:
            O: (batch, N_q, d)
        """
        batch, N_q, d = Q.shape
        _, N_k, _ = K.shape
        scale = 1.0 / math.sqrt(d)

        # Tile sizes (at least 16, power-of-2 friendly)
        B_q = max(16, min(64, N_q))
        B_k = max(16, min(64, N_k))
        T_q = math.ceil(N_q / B_q)

        # Allocate outputs
        O = torch.empty_like(Q)
        L = torch.empty(batch, N_q, device=Q.device, dtype=torch.float32)

        # Launch kernel with grid (T_q, batch)
        flash_fwd_kernel[(T_q, batch)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            is_causal,
            d,
            B_q,
            B_k,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError(
            "FlashAttention-2 Triton backward pass is not implemented yet."
        )