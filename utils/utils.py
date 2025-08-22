from typing import List


def chunk_text(text: str, language: str, max_length: int) -> List[str]:
    """
    Split text into chunks by sentence delimiter appropriate for the language,
    each chunk ≤ max_length characters.
    """
    delim_map = {"hi": "।", "en": "."}
    delim = delim_map.get(language, ".")
    # ensure we keep the delimiter on each split
    sentences = [s.strip() + delim for s in text.strip().split(delim) if s.strip()]
    chunks: List[str] = []
    current = ""
    for s in sentences:
        # +1 for possible space
        if len(current) + len(s) + 1 <= max_length:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks