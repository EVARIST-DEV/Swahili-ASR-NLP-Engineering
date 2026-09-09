def exact_match(references, predictions):
    refs = [str(x).strip().lower() for x in references]
    preds = [str(x).strip().lower() for x in predictions]
    correct = sum(r == p for r, p in zip(refs, preds))
    total = len(refs)
    return {
        "exact_match": correct / total if total else 0.0,
        "correct": correct,
        "incorrect": total - correct,
    }
