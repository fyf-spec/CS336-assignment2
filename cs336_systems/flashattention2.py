import math

import torch
import einops


@torch.compile
def _flash_attn_backward(Q, K, V, O, dO, L, is_causal=False):
    """
    FlashAttention-2 backward pass with recomputation (Equations 13-19).
    Recomputes P from Q, K, L instead of storing the N×N attention matrix.
    """
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # Eq 13: S = QK^T / √d
    S = torch.bmm(Q, K.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if is_causal:
        N_q, N_k = Q.shape[1], K.shape[1]
        q_idx = torch.arange(N_q, device=Q.device).unsqueeze(1)
        k_idx = torch.arange(N_k, device=Q.device).unsqueeze(0)
        causal_mask = q_idx >= k_idx
        S = torch.where(causal_mask.unsqueeze(0), S, torch.tensor(-1e6, device=S.device, dtype=S.dtype))

    # Eq 14: P = exp(S - L)
    P = torch.exp(S - L.unsqueeze(-1))

    # D = rowsum(O ⊙ dO)
    D = (O * dO).sum(dim=-1)

    # Eq 15: dV = P^T @ dO
    dV = torch.bmm(P.transpose(-2, -1), dO)

    # Eq 16: dP = dO @ V^T
    dP = torch.bmm(dO, V.transpose(-2, -1))

    # Eq 17: dS = P ⊙ (dP - D)
    dS = P * (dP - D.unsqueeze(-1))

    # Eq 18: dQ = dS @ K / √d
    dQ = torch.bmm(dS, K) * scale

    # Eq 19: dK = dS^T @ Q / √d
    dK = torch.bmm(dS.transpose(-2, -1), Q) * scale

    return dQ, dK, dV


class FlashAttentionPytorch(torch.autograd.Function):
    """Pure PyTorch implementation of FlashAttention-2 (Algorithm 1)."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Args:
            Q: (batch, N_q, d)
            K: (batch, N_k, d)
            V: (batch, N_k, d)
            is_causal: bool
        Returns:
            O: (batch, N_q, d)
        """
        # get the dimension of Q, K, V
        batch, N_q, d = Q.shape
        _, N_k, _ = K.shape

        # get the scale of S
        scale = 1.0 / math.sqrt(d)

        # get the num of tiles
        B_q = max(16, min(N_q, 64))
        B_k = max(16, min(N_k, 64))
        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)

        O = torch.zeros(batch, N_q, d, device=Q.device, dtype=Q.dtype)
        L = torch.zeros(batch, N_q, device=Q.device, dtype=Q.dtype)

        for t_q in range(T_q):
            q_start = t_q * B_q
            q_end = min((t_q + 1) * B_q, N_q)
            B_q_actual = q_end - q_start
            Q_i = Q[:, q_start:q_end, :]

            # Initialize O_i(0)=0, l_i(0)=0, m_i(0)=-inf
            O_i = torch.zeros(batch, B_q_actual, d, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(batch, B_q_actual, device=Q.device, dtype=Q.dtype)
            m_i = torch.full((batch, B_q_actual), float("-inf"), device=Q.device, dtype=Q.dtype)

            for t_k in range(T_k):
                k_start = t_k * B_k
                k_end = min((t_k + 1) * B_k, N_k)

                # Load K_j, V_j from global memory
                K_i = K[:, k_start:k_end, :]
                V_i = V[:, k_start:k_end, :]
                # Compute tile of pre-softmax attention
                S_i = einops.einsum(Q_i, K_i, "b i d, b j d -> b i j") * scale # S_i:(batch, B_q, B_k)
                # Compute m_i(j)
                m_i_next = torch.max(m_i, S_i.max(dim=-1).values)   # m_i:(batch, B_q)
                # Compute P_i(j)
                P_i = torch.exp(S_i - m_i_next.unsqueeze(-1))   # P_i : (batch, B_q, B_k)
                # Compute l_i(j)
                l_i = torch.exp(m_i - m_i_next) * l_i + torch.sum(P_i, dim=-1) # l_i:(B_q)
                # Compute O_i(j) = diag(exp(m_i - m_i_next)) * O_i(j-1) +  P_i(j)V_i(j)
                O_i = torch.exp(m_i - m_i_next).unsqueeze(-1) * O_i + einops.einsum(P_i, V_i, "b i j, b j d -> b i d") # O_i:(B_q, d)
                # Update m_i(j)
                m_i = m_i_next
            O_i = O_i / l_i.unsqueeze(-1) #O_i:(batch, B_q, d)
            L_i = m_i + torch.log(l_i) # L_i:(batch, B_q)

            O[:, q_start:q_end, :] = O_i
            L[:, q_start:q_end] = L_i

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_attn_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None  # None for is_causal