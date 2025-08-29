from typing import List
import numpy as np
from scipy.signal import butter, filtfilt


def chunk_text(text: str, language: str, max_length: int) -> List[str]:
    """
    Split text into chunks by words (space), each chunk ≤ max_length characters.
    Words are not split between chunks.
    """
    words = text.strip().split()
    chunks: List[str] = []
    current_words = []
    current_len = 0

    for word in words:
        add_len = len(word) + (1 if current_words else 0)
        if current_len + add_len <= max_length:
            current_words.append(word)
            current_len += add_len
        else:
            if current_words:
                chunks.append(' '.join(current_words))
            # Start new chunk with current word
            current_words = [word]
            current_len = len(word)
    if current_words:
        chunks.append(' '.join(current_words))
    return chunks

def create_silence_padding(sample_rate: int = 24000, duration_ms: int = 100) -> np.ndarray:
    """Create a silence padding of specified duration in milliseconds"""
    num_samples = int((duration_ms / 1000) * sample_rate)
    return np.zeros(num_samples)

def crossfade(a: np.ndarray, b: np.ndarray, overlap_samples: int = 1000) -> np.ndarray:
    """Crossfade two audio segments"""
    if len(a) < overlap_samples or len(b) < overlap_samples:
        return np.concatenate([a, b])
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, overlap_samples)
    fade_in = np.linspace(0.0, 1.0, overlap_samples)
    
    # Apply crossfade
    a[-overlap_samples:] *= fade_out
    b[:overlap_samples] *= fade_in
    
    return np.concatenate([a[:-overlap_samples], b])

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to prevent volume differences"""
    return audio / (np.max(np.abs(audio)) + 1e-6)

def apply_lowpass(audio: np.ndarray, cutoff: float = 10000, fs: int = 24000) -> np.ndarray:
    """Apply lowpass filter to reduce high-frequency artifacts"""
    nyquist = fs * 0.5
    normal_cutoff = cutoff / nyquist
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, audio)
