"""Post-hoc explainer baselines: LIME and Integrated Gradients."""

import numpy


def make_lime_explain_fn(predictor, num_samples: int = 500, seed: int = 42):
    """Create a LIME explain function compatible with the benchmark contract.

    Args:
        predictor: Object with predict_proba(texts) -> list[list[float]].
        num_samples: Number of perturbation samples for LIME.
        seed: Random seed for reproducibility.

    Returns:
        Callable (text, top_k) -> list[tuple[str, float]].
    """
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(random_state=seed)

    def explain_fn(text: str, top_k: int) -> list[tuple[str, float]]:
        def predict_fn(texts_list):
            return numpy.array(predictor.predict_proba(list(texts_list)))

        exp = explainer.explain_instance(
            text, predict_fn, num_features=top_k, num_samples=num_samples
        )
        return exp.as_list()

    return explain_fn


def make_ig_explain_fn(model, tokenizer, max_length: int, n_steps: int = 50):
    """Create an Integrated Gradients explain function via captum.

    Uses LayerIntegratedGradients on the embedding layer. Returns subword-level
    attributions that need normalize_attributions_to_words for word-level mapping.

    Args:
        model: SpladeModel (possibly torch.compiled).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        n_steps: Number of interpolation steps for IG.

    Returns:
        Callable (text, top_k) -> list[tuple[str, float]].
    """
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

        # Baseline: PAD token embeddings (zero-information input)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

        # Determine target class
        with torch.inference_mode():
            logits, _ = _model(input_ids, attention_mask)
        target = int(logits.argmax(dim=-1).item())

        # Compute IG attributions (requires gradients — captum handles this)
        attrs = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=n_steps,
        )
        # attrs: [1, seq_len, hidden_dim] — sum over embedding dim for per-token score
        token_attrs = attrs.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        # Map to tokens, skip special tokens and padding
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        scored = [
            (tok, float(token_attrs[i]))
            for i, tok in enumerate(tokens)
            if tok not in SPECIAL_TOKENS and attention_mask[0][i].item() == 1
        ]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:top_k]

    return explain_fn
