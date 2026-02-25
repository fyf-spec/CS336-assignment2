"""Monkey-patch all model / nn_utils functions with NVTX range annotations.

Usage:
    import cs336_basics.nvtx_wrapper  # patches happen at import time
"""

import torch
import cs336_basics.model as _model
import cs336_basics.nn_utils as _nn_utils


def _nvtx_wrap(name, fn):
    """Return a wrapper that pushes/pops an NVTX range around *fn*."""
    def wrapper(*args, **kwargs):
        torch.cuda.nvtx.range_push(name)
        result = fn(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return result
    return wrapper


# ── Standalone functions in cs336_basics.model ────────────────────────────────

_orig_scaled_dot_product_attention = _model.scaled_dot_product_attention
_model.scaled_dot_product_attention = _nvtx_wrap(
    "scaled_dot_product_attention", _orig_scaled_dot_product_attention
)

_orig_softmax = _model.softmax
_model.softmax = _nvtx_wrap("softmax", _orig_softmax)

_orig_silu = _model.silu
_model.silu = _nvtx_wrap("silu", _orig_silu)


# ── Class .forward methods in cs336_basics.model ──────────────────────────────

_orig_TransformerLM_forward = _model.TransformerLM.forward
_model.TransformerLM.forward = _nvtx_wrap(
    "TransformerLM.forward", _orig_TransformerLM_forward
)

_orig_TransformerBlock_forward = _model.TransformerBlock.forward
_model.TransformerBlock.forward = _nvtx_wrap(
    "TransformerBlock.forward", _orig_TransformerBlock_forward
)

_orig_MultiHeadSelfAttentionWithRope_forward = _model.MultiHeadSelfAttentionWithRope.forward
_model.MultiHeadSelfAttentionWithRope.forward = _nvtx_wrap(
    "MultiHeadSelfAttentionWithRope.forward",
    _orig_MultiHeadSelfAttentionWithRope_forward,
)

_orig_RotaryPositionalEmbedding_forward = _model.RotaryPositionalEmbedding.forward
_model.RotaryPositionalEmbedding.forward = _nvtx_wrap(
    "RotaryPositionalEmbedding.forward", _orig_RotaryPositionalEmbedding_forward
)

_orig_RMSNorm_forward = _model.RMSNorm.forward
_model.RMSNorm.forward = _nvtx_wrap("RMSNorm.forward", _orig_RMSNorm_forward)

_orig_SwiGLU_forward = _model.SwiGLU.forward
_model.SwiGLU.forward = _nvtx_wrap("SwiGLU.forward", _orig_SwiGLU_forward)

_orig_Linear_forward = _model.Linear.forward
_model.Linear.forward = _nvtx_wrap("Linear.forward", _orig_Linear_forward)

_orig_Embedding_forward = _model.Embedding.forward
_model.Embedding.forward = _nvtx_wrap("Embedding.forward", _orig_Embedding_forward)


# ── Standalone functions in cs336_basics.nn_utils ─────────────────────────────

_orig_cross_entropy = _nn_utils.cross_entropy
_nn_utils.cross_entropy = _nvtx_wrap("cross_entropy", _orig_cross_entropy)

_orig_log_softmax = _nn_utils.log_softmax
_nn_utils.log_softmax = _nvtx_wrap("log_softmax", _orig_log_softmax)
