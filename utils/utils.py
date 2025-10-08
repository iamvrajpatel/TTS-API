from typing import List
import numpy as np
from scipy.signal import butter, filtfilt
import re
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Request
import asyncio
import os

def chunk_text(text: str, language: str, max_length: int = 230) -> List[str]:
    """
    Split text into sentence-based chunks, each ≤ max_length characters.
    If a single sentence exceeds max_length, it will be further split by words.
    """
    # 1. Split into sentences
    sentences = re.split(r'(?<=[.!?।:;,\-])\s+', text.strip())

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        # If sentence itself is longer than max_length → split by words
        if len(sentence) > max_length:
            words = sentence.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_length:
                    temp_chunk += (" " if temp_chunk else "") + word
                else:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = word
            if temp_chunk:
                chunks.append(temp_chunk.strip())
            continue  # move to next sentence

        # Otherwise, try to add the sentence to the current chunk
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

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


async def clone_voice_chunks(text_chunks, cleaned_ref_audio_path, language, request: Request):
    audio_chunks = []
    for i, chunk in enumerate(text_chunks):
        if await request.is_disconnected():
            raise asyncio.CancelledError()
        try:
            # ✅ FIX: remove overlap logic, just use clean chunk
            wav = await asyncio.to_thread(
                tts.tts,
                text=chunk,
                language=language,
                speaker_wav=cleaned_ref_audio_path
            )
            wav = normalize_audio(wav)
            wav = apply_lowpass(wav)
            audio_chunks.append(wav)
        except Exception as e:
            raise HTTPException(500, f"TTS generation failed on chunk '{chunk}': {e}")
    return audio_chunks

async def run_tts_task(func, *args, **kwargs):
    # Acquire semaphore for limited concurrency
    async with tts_semaphore:
        # Check for cancellation
        try:
            return await func(*args, **kwargs)
        except asyncio.CancelledError:
            # Optionally cleanup resources here
            raise HTTPException(499, "Request cancelled by client.")

async def synthesize_tts_chunks(chunks, ref_wav, lang, request: Request):
    waves: List[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        if await request.is_disconnected():
            raise asyncio.CancelledError()
        try:
            # ✅ FIX: do not add overlap words from next chunk
            wav = await asyncio.to_thread(
                tts.tts,
                text=chunk,
                speaker_wav=ref_wav,
                language=lang
            )
            wav = normalize_audio(wav)
            wav = apply_lowpass(wav)
            waves.append(wav)
        except Exception as e:
            raise HTTPException(500, f"TTS generation failed on chunk '{chunk}': {e}")
    return waves