def _clean_subword(token: str) -> str:
    if token.startswith("##"):
        return token[2:]
    if token.startswith("\u2581"):
        return token[1:]
    if token.startswith("\u0120"):
        return token[1:]
    return token


def normalize_attributions_to_words(
    text: str,
    attrib: list[tuple[str, float]],
    tokenizer,
) -> list[tuple[str, float]]:
    if not attrib:
        return []

    subword_scores: dict[str, float] = {}
    for token, score in attrib:
        clean = _clean_subword(token.lower())
        if clean and clean not in subword_scores:
            subword_scores[clean] = score

    words = text.split()
    word_scores: list[tuple[str, float]] = []

    for word in words:
        subwords = tokenizer.tokenize(word)
        if not subwords:
            continue
        total_abs = 0.0
        dominant_score = 0.0
        has_match = False
        for sw in subwords:
            clean_sw = _clean_subword(sw.lower())
            if clean_sw in subword_scores:
                score = subword_scores[clean_sw]
                total_abs += abs(score)
                if abs(score) > abs(dominant_score):
                    dominant_score = score
                has_match = True
        if has_match:
            sign = 1.0 if dominant_score >= 0 else -1.0
            word_scores.append((word, sign * total_abs))

    word_scores.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return word_scores
