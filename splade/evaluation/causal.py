"""Causal faithfulness metrics via MLM-driven counterfactual generation.

Uses a Masked Language Model to generate contextually valid counterfactuals
by masking salient tokens and sampling plausible alternatives.
"""

import re
from collections import defaultdict

import numpy
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE

class MLMCounterfactualGenerator:
    """Generates counterfactuals using a Masked Language Model."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation="sdpa").to(DEVICE)
        self.model.eval()

    def generate(self, text: str, target_word: str, top_k: int = 5) -> str | None:
        """Replace `target_word` with a contextually plausible MLM-predicted alternative."""
        word_tokens = self.tokenizer.encode(target_word, add_special_tokens=False)
        if not word_tokens:
            return None

        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"][0]

        # Sliding window search for the subword sequence
        seq_len = len(word_tokens)
        match_idx = -1
        word_tensor = torch.tensor(word_tokens, device=DEVICE)
        for i in range(len(input_ids) - seq_len + 1):
            if torch.equal(input_ids[i : i + seq_len], word_tensor):
                match_idx = i
                break

        if match_idx == -1:
            return None

        # Mask all tokens of the target word
        masked_input_ids = input_ids.clone()
        masked_input_ids[match_idx : match_idx + seq_len] = self.tokenizer.mask_token_id

        # Predict at each masked position independently
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            outputs = self.model(masked_input_ids.unsqueeze(0))
            replacement_ids = []
            for offset in range(seq_len):
                pos = match_idx + offset
                logits = outputs.logits[0, pos]
                probs = F.softmax(logits, dim=-1)
                top_ids = torch.topk(probs, k=top_k + 1).indices
                # Pick the best token that differs from the original at this position
                chosen = top_ids[0]
                for tid in top_ids:
                    if tid.item() != word_tokens[offset]:
                        chosen = tid
                        break
                replacement_ids.append(chosen.item())

        new_word = self.tokenizer.decode(replacement_ids).strip()
        if not new_word or new_word.lower() == target_word.lower():
            return None

        return re.sub(r'\b' + re.escape(target_word) + r'\b', new_word, text, count=1)

def compute_causal_faithfulness(
    model,
    tokenizer,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    max_length: int,
    generator: MLMCounterfactualGenerator | None = None,
    attribution_levels: tuple[int, ...] = (1, 3, 5),
) -> float:
    """Compute Causal Faithfulness using multi-level MLM Counterfactuals + Spearman."""
    from scipy.stats import spearmanr
    from splade.inference import predict_proba_model

    if generator is None:
        generator = MLMCounterfactualGenerator()

    original_probs = predict_proba_model(model, tokenizer, texts, max_length)

    # Phase 1: Generate all counterfactuals, collecting texts for batched prediction
    counterfactual_tasks = []  # (text_idx, level, target_class, avg_score, cf_text)
    for i, text in enumerate(texts):
        if not attributions[i]:
            continue
        target = int(numpy.argmax(original_probs[i]))
        for level in attribution_levels:
            top_tokens = [(t, s) for t, s in attributions[i][:level] if s > 0]
            if not top_tokens:
                continue
            avg_score = float(numpy.mean([abs(s) for _, s in top_tokens]))
            for token, _ in top_tokens:
                new_text = generator.generate(text, token)
                if new_text is not None:
                    counterfactual_tasks.append((i, level, target, avg_score, new_text))

    if not counterfactual_tasks:
        return float('nan')

    # Phase 2: Batch predict all counterfactual texts at once
    cf_texts = [task[4] for task in counterfactual_tasks]
    all_cf_probs = predict_proba_model(model, tokenizer, cf_texts, max_length)

    # Phase 3: Aggregate shifts per (text_idx, level)
    level_data = defaultdict(lambda: {"shifts": [], "avg_score": 0.0})
    for task_idx, (text_idx, level, target, avg_score, _) in enumerate(counterfactual_tasks):
        key = (text_idx, level)
        orig_conf = original_probs[text_idx][target]
        cf_conf = all_cf_probs[task_idx][target]
        level_data[key]["shifts"].append(abs(orig_conf - cf_conf))
        level_data[key]["avg_score"] = avg_score

    shifts = []
    attrib_scores = []
    for key, data in level_data.items():
        if data["shifts"]:
            shifts.append(float(numpy.mean(data["shifts"])))
            attrib_scores.append(data["avg_score"])

    if len(shifts) < 3:
        return float('nan')
    corr, _ = spearmanr(attrib_scores, shifts)
    return float(corr) if not numpy.isnan(corr) else float('nan')
