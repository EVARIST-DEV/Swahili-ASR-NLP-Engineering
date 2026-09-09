import re
import unicodedata

def normalize_swahili_text(text: str) -> str:
    """Normalize Swahili text for consistent ASR evaluation."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
