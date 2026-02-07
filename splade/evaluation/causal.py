"""Causal faithfulness metrics via MLM-driven counterfactual generation.

Uses a Masked Language Model to generate contextually valid counterfactuals
by masking salient tokens and sampling plausible alternatives.
"""

import re

import numpy
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.utils.cuda import DEVICE

class MLMCounterfactualGenerator:
    """Generates counterfactuals using a Masked Language Model."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def generate(self, text: str, target_word: str, top_k: int = 5) -> str | None:
        """
        Replace `target_word` in `text` with a contextually plausible alternative 
        predicted by the MLM.
        """
        # 1. Tokenize and find position of target word
        # This is heuristic; precise alignment is hard without span info.
        # We search for the first occurrence of the word.
        
        # Naive tokenization check
        # We need to ensure we mask the *correct* tokens corresponding to the word.
        word_tokens = self.tokenizer.encode(target_word, add_special_tokens=False)
        if not word_tokens:
            return None
            
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"][0]
        
        # Search for the sequence of word_tokens
        # Sliding window search
        seq_len = len(word_tokens)
        match_idx = -1
        for i in range(len(input_ids) - seq_len + 1):
            if torch.equal(input_ids[i : i + seq_len], torch.tensor(word_tokens, device=DEVICE)):
                match_idx = i
                break
                
        if match_idx == -1:
            # Fallback: try case-insensitive
            # (Skipping complex alignment for efficiency in this simplified impl)
            return None

        # 2. Mask the tokens
        masked_input_ids = input_ids.clone()
        masked_input_ids[match_idx : match_idx + seq_len] = self.tokenizer.mask_token_id
        
        # 3. Predict
        with torch.no_grad():
            outputs = self.model(masked_input_ids.unsqueeze(0))
            logits = outputs.logits[0, match_idx] # Logits at the mask position (using first mask token if multi-token)
            
        # 4. Sample
        # We want a word that is NOT the original word.
        probs = F.softmax(logits, dim=-1)
        top_ids = torch.topk(probs, k=top_k + 1).indices
        
        for idx in top_ids:
            if idx not in word_tokens: # Check if it's different
                # Decode the new token
                new_word = self.tokenizer.decode([idx]).strip()
                # Basic filter: ignore subwords or special chars if possible
                if new_word.isalnum() and new_word.lower() != target_word.lower():
                    # Construct new text
                    # We replace the tokens in the input_ids and decode back
                    new_input_ids = input_ids.clone()
                    # For multi-token replacement, this is tricky. 
                    # Simpler: replace in string if we trust the word.
                    # Or replace just the first token and drop the others?
                    # Let's simple string replace for stability.
                    return re.sub(r'\b' + re.escape(target_word) + r'\b', new_word, text, count=1)
                    
        return None

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
    shifts = []
    attrib_scores = []

    for i, text in enumerate(texts):
        if not attributions[i]:
            continue
        target = numpy.argmax(original_probs[i])
        for level in attribution_levels:
            top_tokens = [(t, s) for t, s in attributions[i][:level] if s > 0]
            if not top_tokens:
                continue
            avg_score = numpy.mean([abs(s) for _, s in top_tokens])
            level_shifts = []
            for token, _ in top_tokens:
                new_text = generator.generate(text, token)
                if new_text is not None:
                    new_prob = predict_proba_model(model, tokenizer, [new_text], max_length)[0]
                    level_shifts.append(abs(original_probs[i][target] - new_prob[target]))
            if level_shifts:
                shifts.append(numpy.mean(level_shifts))
                attrib_scores.append(avg_score)

    if len(shifts) < 3:
        return float('nan')
    corr, _ = spearmanr(attrib_scores, shifts)
    return float(corr) if not numpy.isnan(corr) else float('nan')
