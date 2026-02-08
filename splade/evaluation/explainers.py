"""Explainer baselines for interpretability benchmarking.

Provides factory functions for: LIME, Integrated Gradients, GradientSHAP,
Attention Rollout, Saliency (Gradient x Input), and DeepLIFT.
All use the (text, top_k) -> list[tuple[str, float]] interface.
"""

import numpy

from splade.evaluation.constants import (
    LIME_N_SAMPLES,
    IG_N_STEPS,
    GRADIENT_SHAP_N_SAMPLES,
)


def make_lime_explain_fn(predictor, seed: int = 42):
    """LIME text explainer (Ribeiro et al., 2016)."""
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(random_state=seed)

    def explain_fn(text: str, top_k: int) -> list[tuple[str, float]]:
        def predict_fn(texts_list):
            return numpy.array(predictor.predict_proba(list(texts_list)))

        exp = explainer.explain_instance(
            text, predict_fn, num_features=top_k, num_samples=LIME_N_SAMPLES
        )
        return exp.as_list()

    return explain_fn


def make_ig_explain_fn(model, tokenizer, max_length: int):
    """Integrated Gradients (Sundararajan et al., 2017) via captum."""
    import torch
    from captum.attr import LayerIntegratedGradients
    from splade.utils.cuda import DEVICE
    from splade.inference import SPECIAL_TOKENS

    _model = model._orig_mod if hasattr(model, "_orig_mod") else model

    def _forward_for_captum(input_ids, attention_mask):
        logits, _ = _model(input_ids, attention_mask)
        return logits

    lig = LayerIntegratedGradients(_forward_for_captum, _model.bert.embeddings)

    def explain_fn(text: str, top_k: int) -> list[tuple[str, float]]:
        encoding = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

        with torch.inference_mode():
            logits, _ = _model(input_ids, attention_mask)
        target = int(logits.argmax(dim=-1).item())

        attrs = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=IG_N_STEPS,
        )
        token_attrs = attrs.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        attn_mask_cpu = attention_mask[0].cpu().tolist()
        scored = [
            (tok, float(token_attrs[i]))
            for i, tok in enumerate(tokens)
            if tok not in SPECIAL_TOKENS and attn_mask_cpu[i] == 1
        ]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:top_k]

    return explain_fn


def make_gradient_shap_explain_fn(model, tokenizer, max_length: int):
    """GradientSHAP (Lundberg & Lee, 2017) via captum."""
    import torch
    from captum.attr import LayerGradientShap
    from splade.utils.cuda import DEVICE
    from splade.inference import SPECIAL_TOKENS

    _model = model._orig_mod if hasattr(model, "_orig_mod") else model

    def _forward_for_captum(input_ids, attention_mask):
        logits, _ = _model(input_ids, attention_mask)
        return logits

    lgs = LayerGradientShap(_forward_for_captum, _model.bert.embeddings)

    def explain_fn(text: str, top_k: int) -> list[tuple[str, float]]:
        encoding = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        # Stochastic baselines: zero + random PAD permutations
        baselines = torch.full(
            (GRADIENT_SHAP_N_SAMPLES, input_ids.shape[1]),
            tokenizer.pad_token_id,
            dtype=input_ids.dtype,
            device=DEVICE,
        )

        with torch.inference_mode():
            logits, _ = _model(input_ids, attention_mask)
        target = int(logits.argmax(dim=-1).item())

        attrs = lgs.attribute(
            inputs=input_ids,
            baselines=baselines,
            additional_forward_args=(attention_mask,),
            target=target,
            n_samples=GRADIENT_SHAP_N_SAMPLES,
        )
        token_attrs = attrs.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        attn_mask_cpu = attention_mask[0].cpu().tolist()
        scored = [
            (tok, float(token_attrs[i]))
            for i, tok in enumerate(tokens)
            if tok not in SPECIAL_TOKENS and attn_mask_cpu[i] == 1
        ]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:top_k]

    return explain_fn


def make_attention_explain_fn(model, tokenizer, max_length: int):
    """Attention Rollout (Abnar & Zuidema, 2020).

    Recursively multiplies attention matrices across layers to compute
    total attention flow from input tokens to the [CLS] token.
    """
    import torch
    from splade.utils.cuda import DEVICE, COMPUTE_DTYPE
    from splade.inference import SPECIAL_TOKENS

    _model = model._orig_mod if hasattr(model, "_orig_mod") else model

    def explain_fn(text: str, top_k: int) -> list[tuple[str, float]]:
        encoding = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            outputs = _model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Attention rollout: multiply attention matrices across layers
        # Each attention tensor: [batch, heads, seq, seq]
        attentions = outputs.attentions
        rollout = None
        for attn in attentions:
            # Average over attention heads
            attn_mean = attn.mean(dim=1)  # [batch, seq, seq]
            # Add identity (residual connection)
            identity = torch.eye(attn_mean.shape[-1], device=attn_mean.device)
            attn_with_residual = (attn_mean + identity) / 2
            # Renormalize rows
            attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
            if rollout is None:
                rollout = attn_with_residual
            else:
                rollout = torch.bmm(attn_with_residual, rollout)

        # Extract attention from [CLS] (position 0) to all other tokens
        cls_attention = rollout[0, 0].cpu().numpy()  # [seq_len]

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        attn_mask_cpu = attention_mask[0].cpu().tolist()
        scored = [
            (tok, float(cls_attention[i]))
            for i, tok in enumerate(tokens)
            if tok not in SPECIAL_TOKENS and attn_mask_cpu[i] == 1
        ]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:top_k]

    return explain_fn


def make_saliency_explain_fn(model, tokenizer, max_length: int):
    """Gradient x Input saliency (Simonyan et al., 2014) via captum."""
    import torch
    from captum.attr import InputXGradient
    from splade.utils.cuda import DEVICE
    from splade.inference import SPECIAL_TOKENS

    _model = model._orig_mod if hasattr(model, "_orig_mod") else model

    def _forward_from_embeds(embeddings, attention_mask):
        logits, _ = _model.forward_from_embeddings(embeddings, attention_mask)
        return logits

    ixg = InputXGradient(_forward_from_embeds)

    def explain_fn(text: str, top_k: int) -> list[tuple[str, float]]:
        encoding = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        # Get embeddings (requires grad for saliency)
        embeddings = _model.get_embeddings(input_ids).detach().requires_grad_(True)

        with torch.inference_mode():
            logits, _ = _model(input_ids, attention_mask)
        target = int(logits.argmax(dim=-1).item())

        attrs = ixg.attribute(
            inputs=embeddings,
            additional_forward_args=(attention_mask,),
            target=target,
        )
        # Sum over embedding dim for per-token score
        token_attrs = attrs.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        attn_mask_cpu = attention_mask[0].cpu().tolist()
        scored = [
            (tok, float(token_attrs[i]))
            for i, tok in enumerate(tokens)
            if tok not in SPECIAL_TOKENS and attn_mask_cpu[i] == 1
        ]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:top_k]

    return explain_fn


def make_deeplift_explain_fn(model, tokenizer, max_length: int):
    """DeepLIFT (Shrikumar et al., 2017) via captum.

    Uses LayerDeepLift on the embedding layer with PAD token baseline.
    """
    import torch
    from captum.attr import LayerDeepLift
    from splade.utils.cuda import DEVICE
    from splade.inference import SPECIAL_TOKENS

    _model = model._orig_mod if hasattr(model, "_orig_mod") else model

    def _forward_for_captum(input_ids, attention_mask):
        logits, _ = _model(input_ids, attention_mask)
        return logits

    ldl = LayerDeepLift(_forward_for_captum, _model.bert.embeddings)

    def explain_fn(text: str, top_k: int) -> list[tuple[str, float]]:
        encoding = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

        with torch.inference_mode():
            logits, _ = _model(input_ids, attention_mask)
        target = int(logits.argmax(dim=-1).item())

        attrs = ldl.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target,
        )
        token_attrs = attrs.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        attn_mask_cpu = attention_mask[0].cpu().tolist()
        scored = [
            (tok, float(token_attrs[i]))
            for i, tok in enumerate(tokens)
            if tok not in SPECIAL_TOKENS and attn_mask_cpu[i] == 1
        ]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:top_k]

    return explain_fn
