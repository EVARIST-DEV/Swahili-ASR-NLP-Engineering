from jiwer import wer, cer, process_words

def compute_asr_metrics(references, hypotheses):
    return {
        "wer": float(wer(references, hypotheses)),
        "cer": float(cer(references, hypotheses)),
    }

def word_error_breakdown(references, hypotheses):
    alignment = process_words(references, hypotheses)
    return {
        "substitutions": int(alignment.substitutions),
        "deletions": int(alignment.deletions),
        "insertions": int(alignment.insertions),
        "reference_words": int(alignment.hits + alignment.substitutions + alignment.deletions),
        "correct_words": int(alignment.hits),
    }
