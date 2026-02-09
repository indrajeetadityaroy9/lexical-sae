"""Mechanistic evaluation for sequence labelling (NER).

Uses seqeval for standard CoNLL evaluation. Tier 3 circuit examples
explicitly filter O tags and IGNORE_INDEX to show meaningful entity
attributions.
"""

from dataclasses import dataclass, field

import torch
from seqeval.metrics import classification_report, f1_score as seqeval_f1

from splade.data.ner_loader import CONLL2003_LABEL_NAMES, IGNORE_INDEX
from splade.mechanistic.attribution import compute_attribution_tensor
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


@dataclass
class SequenceMechanisticResults:
    token_accuracy: float = 0.0
    entity_f1: float = 0.0
    classification_report: str = ""
    dla_verification_error: float = 0.0
    mean_active_dims: float = 0.0
    num_eval_samples: int = 0


def _predictions_to_tag_lists(
    pred_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_names: list[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert model predictions to seqeval format.

    Filters out IGNORE_INDEX and padding positions. Returns list-of-lists
    of tag strings as required by seqeval.

    Args:
        pred_ids: [B, L] predicted tag indices.
        gold_ids: [B, L] gold tag indices.
        attention_mask: [B, L] attention mask.
        label_names: Tag name list (index -> string).

    Returns:
        (pred_tags, gold_tags): Each is list[list[str]] for seqeval.
    """
    pred_tags = []
    gold_tags = []

    B, L = gold_ids.shape
    for b in range(B):
        pred_seq = []
        gold_seq = []
        for l in range(L):
            if attention_mask[b, l] == 0:
                continue
            gold_label = gold_ids[b, l].item()
            if gold_label == IGNORE_INDEX:
                continue
            pred_seq.append(label_names[pred_ids[b, l].item()])
            gold_seq.append(label_names[gold_label])
        pred_tags.append(pred_seq)
        gold_tags.append(gold_seq)

    return pred_tags, gold_tags


def run_sequence_mechanistic_evaluation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    token_labels: torch.Tensor,
    tokenizer,
    label_names: list[str] | None = None,
    centroid_tracker=None,
) -> SequenceMechanisticResults:
    """Run tiered mechanistic evaluation for sequence labelling.

    Tier 1: Token accuracy, entity F1, DLA verification.
    Tier 2: Mean active dims per position.
    Tier 3: Per-tag circuit examples (O-tags filtered out).

    Args:
        model: LexicalSAE (sequence_labeling).
        input_ids: [N, L] test token IDs.
        attention_masks: [N, L] test attention masks.
        token_labels: [N, L] test labels.
        tokenizer: HuggingFace tokenizer for decoding.
        label_names: Tag name list (defaults to CONLL2003_LABEL_NAMES).
        centroid_tracker: Optional trained TokenAttributionCentroidTracker.

    Returns:
        SequenceMechanisticResults.
    """
    if label_names is None:
        label_names = CONLL2003_LABEL_NAMES

    _model = unwrap_compiled(model)
    results = SequenceMechanisticResults()
    results.num_eval_samples = len(input_ids)

    all_pred_tags = []
    all_gold_tags = []
    total_correct = 0
    total_valid = 0
    total_active_dims = 0.0
    total_dla_error = 0.0
    dla_samples = 0

    eval_batch = 16
    for start in range(0, len(input_ids), eval_batch):
        end = min(start + eval_batch, len(input_ids))
        batch_ids = input_ids[start:end].to(DEVICE)
        batch_mask = attention_masks[start:end].to(DEVICE)
        batch_labels = token_labels[start:end].to(DEVICE)

        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq = _model(batch_ids, batch_mask)
            token_logits = _model.tag(sparse_seq)

        # Token accuracy
        preds = token_logits.argmax(dim=-1)  # [B, L]
        valid = batch_mask.bool() & (batch_labels != IGNORE_INDEX)
        total_correct += (preds[valid] == batch_labels[valid]).sum().item()
        total_valid += valid.sum().item()

        # Active dims per position
        active = (sparse_seq > 0).sum(dim=-1).float()  # [B, L]
        total_active_dims += active[batch_mask.bool()].sum().item()

        # seqeval format
        pred_t, gold_t = _predictions_to_tag_lists(
            preds.cpu(), batch_labels.cpu(), batch_mask.cpu(), label_names,
        )
        all_pred_tags.extend(pred_t)
        all_gold_tags.extend(gold_t)

        # DLA verification (sample up to 64 positions per batch)
        if dla_samples < 512:
            valid_positions = valid.nonzero(as_tuple=False)
            n_sample = min(64, len(valid_positions))
            if n_sample > 0:
                sample_idx = valid_positions[:n_sample]
                sample_sparse = sparse_seq[
                    sample_idx[:, 0], sample_idx[:, 1]
                ]  # [n_sample, V]
                sample_labels = batch_labels[
                    sample_idx[:, 0], sample_idx[:, 1]
                ]  # [n_sample]

                with torch.inference_mode():
                    sample_logits, W_eff, b_eff = _model.classifier_forward(
                        sample_sparse,
                    )

                attr = compute_attribution_tensor(sample_sparse, W_eff, sample_labels)
                actual = sample_logits.gather(
                    1, sample_labels.unsqueeze(1)
                ).squeeze(1)
                reconstructed = attr.sum(dim=-1) + b_eff.gather(
                    1, sample_labels.unsqueeze(1)
                ).squeeze(1)
                total_dla_error += (actual - reconstructed).abs().sum().item()
                dla_samples += n_sample

    results.token_accuracy = total_correct / total_valid if total_valid > 0 else 0.0
    results.mean_active_dims = total_active_dims / total_valid if total_valid > 0 else 0.0
    results.dla_verification_error = total_dla_error / dla_samples if dla_samples > 0 else 0.0

    # seqeval metrics
    results.entity_f1 = seqeval_f1(all_gold_tags, all_pred_tags)
    results.classification_report = classification_report(
        all_gold_tags, all_pred_tags,
    )

    return results


def print_sequence_mechanistic_results(
    results: SequenceMechanisticResults,
    model: torch.nn.Module | None = None,
    tokenizer=None,
    input_ids: torch.Tensor | None = None,
    attention_masks: torch.Tensor | None = None,
    token_labels: torch.Tensor | None = None,
    label_names: list[str] | None = None,
    centroid_tracker=None,
) -> None:
    """Print tiered NER evaluation report.

    Tier 3 explicitly filters O tags to show meaningful entity circuit examples.
    """
    if label_names is None:
        label_names = CONLL2003_LABEL_NAMES

    print("\n" + "=" * 80)
    print("NER SEQUENCE EVALUATION REPORT")
    print("=" * 80)

    # --- Tier 1: Performance ---
    print("\n[Tier 1: Performance]")
    print(f"  Token Accuracy:         {results.token_accuracy:.4f}")
    print(f"  Entity F1 (micro):      {results.entity_f1:.4f}")
    print(f"  DLA Verification Error: {results.dla_verification_error:.6f}")

    tier1_pass = results.dla_verification_error < 0.01
    if not tier1_pass:
        print("\n  !! DLA verification failed (error >= 0.01). Skipping Tiers 2-3.")
        print("=" * 80)
        return

    # --- Tier 2: Faithfulness ---
    print("\n[Tier 2: Faithfulness]")
    print(f"  Mean active dims/pos:   {results.mean_active_dims:.1f}")

    # --- Tier 3: Interpretability (entity circuits) ---
    print("\n[Tier 3: Interpretability]")
    print("\n  seqeval classification report:")
    for line in results.classification_report.strip().split("\n"):
        print(f"    {line}")

    # Show per-tag circuit examples if model + data available
    if (
        model is not None
        and tokenizer is not None
        and input_ids is not None
        and centroid_tracker is not None
    ):
        _model = unwrap_compiled(model)
        print("\n  Entity tag circuit examples (O-tag filtered):")

        # Find examples for each entity tag (skip O = index 0)
        for tag_idx in range(1, len(label_names)):
            tag_name = label_names[tag_idx]

            if (
                not centroid_tracker._initialized[tag_idx]
            ):
                print(f"    {tag_name}: (no trained centroid)")
                continue

            centroid = centroid_tracker.centroids[tag_idx]
            top_vocab_ids = centroid.abs().topk(5).indices.tolist()
            top_scores = centroid.abs().topk(5).values.tolist()
            token_strs = [
                f"{tokenizer.decode([vid]).strip()}({score:.3f})"
                for vid, score in zip(top_vocab_ids, top_scores)
            ]
            print(f"    {tag_name}: {', '.join(token_strs)}")

    print("\n" + "=" * 80)
