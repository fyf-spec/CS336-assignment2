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

@triton.jit
def flash_bwd_kernel_KV(
    Q_ptr, K_ptr, V_ptr, dK_ptr, dV_ptr, O_ptr, dO_ptr, L_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    # In this kernel we don't calculate dQ, because each inner for loop of (T_q) will load and modify Q_tile
    # if we want accurately update dQ, we need to set dQ updating as an atomic step, which will slow the outer loop (T_k)
    # so here we only update dK and dV
    """
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0), 
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_index * stride_db,
        shape=(N_QUERIES, ),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    ) 

    # load k v for outer loop
    K_j = tl.load(K_block_ptr)
    V_j = tl.load(V_block_ptr)
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    for t_q in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        L_i = tl.load(L_block_ptr)
        D_i = tl.load(D_block_ptr)

        Q_i = tl.load(Q_block_ptr)
        O_i = tl.load(O_block_ptr)
        dO_i = tl.load(dO_block_ptr)

        # compute S_i = Q_i @ K_j^T * scale
        S_i = tl.dot(Q_i, tl.trans(K_j)) * scale
        # apply causal mask if needed
        if is_causal:
            q_offsets = t_q * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            S_i = tl.where(causal_mask, S_i, float("-inf"))
        # compute P_i = exp(S_i - L_i)  !!! only for dK and dV !!!
        P_i = tl.exp(S_i - L_i[:,None])
        # compute dV_j += P_i^T @ dO_i
        dV_j += tl.dot(tl.trans(P_i.to(V_j.dtype)), dO_i.to(V_j.dtype))
        # compute dP_i = dO_i @ V_j^T
        dP_i = tl.dot(dO_i, tl.trans(V_j))
        # compute dS_i = P_j * (dP_i - D_i) * scale
        dS_i = P_i * (dP_i - D_i[:,None]) * scale
        # compute dK_j += dS_i^T @ Q_i
        dK_j += tl.dot(tl.trans(dS_i.to(K_j.dtype)), Q_i.to(K_j.dtype))
        
        # advance block ptr for all the inner loop parameters
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))

    # store dK_j and dV_j
    tl.store(dK_block_ptr, dK_j.to(dK_block_ptr.type.element_ty))
    tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty))

@triton.jit
def flash_bwd_kernel_Q(
    Q_ptr, K_ptr, V_ptr, dQ_ptr, O_ptr, dO_ptr, L_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    To avoid atomic updating of dQ, we go through the backward pass again(partially) by recomputing P for dQ
    The outer loop is Q_TILE_SIZE dominated, rather than K_TILE_SIZE dominated in the dK, dV updating path
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0), 
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_index * stride_db,
        shape=(N_QUERIES, ),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    ) 
    
    Q_i = tl.load(Q_block_ptr)
    O_i = tl.load(O_block_ptr)
    dO_i = tl.load(dO_block_ptr)
    L_i = tl.load(L_block_ptr)
    D_i = tl.load(D_block_ptr)

    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for t_k in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        # compute S_j = Q_i @ K_j^T * scale
        S_j = tl.dot(Q_i, tl.trans(K_j)) * scale
        # apply causal mask if needed
        if is_causal:
            k_offsets = t_k * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            S_j = tl.where(causal_mask, S_j, float("-inf"))
        # compute P_j = exp(S_j - L_i)
        P_j = tl.exp(S_j - L_i[:,None])
        # compute dP_j = dO_i @ V_j^T
        dP_j = tl.dot(dO_i, tl.trans(V_j))
        # compute dS_j = P_j * (dP_j - D_i) * scale
        dS_j = P_j * (dP_j - D_i[:,None]) * scale
        # compute dQ_i += dS_j @ K_j
        dQ_i += tl.dot(dS_j.to(K_j.dtype), K_j)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        
    # store dQ
    tl.store(dQ_block_ptr, dQ_i.to(dQ_block_ptr.type.element_ty))

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

        # Tile sizes — must stay within shared memory limits.
        # Shared mem ≈ (B_q + B_k) * D * elem_size  (plus S-tile etc.)
        # Target ≤ ~48 KB per tile pair to stay under 100 KB HW limit.
        elem_size = Q.element_size()  # 2 for bf16/fp16, 4 for fp32
        max_tile = 64
        tile = max_tile
        while tile > 16 and (tile * d * elem_size * 3) > 49152:
            tile //= 2
        B_q = max(16, min(tile, N_q))
        B_k = max(16, min(tile, N_k))
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
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        """
        args:
            dO: (batch, N_q, d)
        returns:
            dQ: (batch, N_q, d)
            dK: (batch, N_k, d)
            dV: (batch, N_k, d)
            None for is_causal
        """
        L, Q, K, V, O = ctx.saved_tensors

        batch, N_q, d = dO.shape
        _, N_k, _ = K.shape
        scale = 1.0 / math.sqrt(d)
        
        # Tile sizes — must stay within shared memory limits.
        # Shared mem ≈ (B_q + B_k) * D * elem_size  (plus S-tile etc.)
        # Target ≤ ~48 KB per tile pair to stay under 100 KB HW limit.
        elem_size = Q.element_size()  # 2 for bf16/fp16, 4 for fp32
        max_tile = 64
        tile = max_tile
        while tile > 16 and (tile * d * elem_size * 3) > 49152:
            tile //= 2
        B_q = max(16, min(tile, N_q))
        B_k = max(16, min(tile, N_k))
        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)
        
        # Calculate D = rowsum(O * dO)
        D = torch.sum(O * dO, dim=-1)
        # Allocate outputs
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        # launch kernels
        flash_bwd_kernel_KV[(T_k, batch)](
            Q, K, V, dK, dV, O, dO, L, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            N_q, N_k,
            scale,
            ctx.is_causal,
            d,
            B_q,
            B_k,
        )
        
        flash_bwd_kernel_Q[(T_q, batch)](
            Q, K, V, dQ, O, dO, L, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            N_q, N_k,
            scale,
            ctx.is_causal,
            d,
            B_q,
            B_k,
        )
        return dQ, dK, dV, None  # None for is_causal