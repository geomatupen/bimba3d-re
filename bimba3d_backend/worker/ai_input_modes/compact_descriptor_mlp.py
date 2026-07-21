"""Public entry points for compact_descriptor_mlp.

This is the 13-input compact MLP comparison model. Keep this wrapper separate
from compact_featurewise_mlp so the simple-input experiment can be removed or
kept independently later.
"""
from bimba3d_backend.worker.ai_input_modes.compact_descriptor_mlp_runtime import (
    predict_compact_descriptor_mlp_multipliers,
    train_compact_descriptor_mlp_model,
)

__all__ = ["predict_compact_descriptor_mlp_multipliers", "train_compact_descriptor_mlp_model"]
