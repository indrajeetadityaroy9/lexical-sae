"""SAE patching: run original + SAE-patched forward passes through the transformer."""

from __future__ import annotations

import torch
from torch import Tensor

from src.data.activation_store import ActivationStore
from src.model.sae import StratifiedSAE
from src.whitening.whitener import SoftZCAWhitener


@torch.no_grad()
def run_patched_forward(
    store: ActivationStore,
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    tokens: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run original and SAE-patched forwards and return logits plus SAE intermediates."""
    model = store.get_model()

    if store._use_transformerlens:
        return _patched_forward_tl(model, sae, whitener, tokens, store.hook_point)
    else:
        return _patched_forward_hf(model, sae, whitener, tokens, store)


def _patched_forward_tl(
    model,
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    tokens: Tensor,
    hook_point: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """TransformerLens: run_with_cache → SAE → run_with_hooks."""
    orig_logits, cache = model.run_with_cache(tokens, names_filter=hook_point)
    x_raw = cache[hook_point]
    B, S, d = x_raw.shape

    x_flat = x_raw.reshape(-1, d).float()
    x_tilde = whitener.forward(x_flat)
    x_hat_flat, z, gate_mask, _ = sae(x_tilde)
    x_hat = x_hat_flat.reshape(B, S, d).to(x_raw.dtype)

    def patch_hook(value, hook):
        return x_hat

    patched_logits = model.run_with_hooks(
        tokens, fwd_hooks=[(hook_point, patch_hook)]
    )

    return orig_logits, patched_logits, x_flat, x_hat_flat, z, gate_mask


def _patched_forward_hf(
    model,
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    tokens: Tensor,
    store: ActivationStore,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """HuggingFace: forward → SAE → hook manipulation → forward."""
    orig_outputs = model(tokens)
    orig_logits = orig_outputs.logits
    x_raw = store._captured_activations
    B, S, d = x_raw.shape

    x_flat = x_raw.reshape(-1, d).float()
    x_tilde = whitener.forward(x_flat)
    x_hat_flat, z, gate_mask, _ = sae(x_tilde)
    x_hat = x_hat_flat.reshape(B, S, d).to(x_raw.dtype)

    original_hook = store._hook_handle
    original_hook.remove()
    layer_idx = store._parse_layer_index()
    target_module = store._resolve_hf_layer(model, layer_idx)

    injected = False

    def inject_hook(module, input, output):
        nonlocal injected
        if not injected:
            injected = True
            if isinstance(output, tuple):
                return (x_hat,) + output[1:]
            return x_hat

    handle = target_module.register_forward_hook(inject_hook)
    patched_outputs = model(tokens)
    patched_logits = patched_outputs.logits
    handle.remove()

    store._register_hf_hook(model)

    return orig_logits, patched_logits, x_flat, x_hat_flat, z, gate_mask
