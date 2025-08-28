import os
import uuid
from datetime import datetime
from typing import List, Tuple
from pydub import AudioSegment

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, constr
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from transformers import GPT2Model
from transformers.generation.utils import GenerationMixin
from scipy.signal import butter, filtfilt

from utils.utils import chunk_text

# Make GPT2Model compatible with GenerationMixin
if not issubclass(GPT2Model, GenerationMixin):
    GPT2Model.__bases__ += (GenerationMixin,)
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs,
])

app = FastAPI(title="4-Way Multilingual TTS (Hindi/English × Male/Female) with Chunking")

# Load on startup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tts: TTS

@app.on_event("startup")
async def load_model():
    global tts
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
              progress_bar=False).to(device)

# Reference voices
VOICE_REFS = {
    "en": {"male": "refs/en_male_ref.mp3", "female": "refs/en_female_ref.mp3"},
    "hi": {"male": "refs/hi_male_ref.mp3", "female": "refs/hi_female_ref.mp3"},
}

SUPPORTED_LANGS   = set(VOICE_REFS.keys())
SUPPORTED_GENDERS = {"male", "female"}

class TTSRequest(BaseModel):
    text:     constr(min_length=1)
    language: constr(to_lower=True)
    gender:   constr(to_lower=True)

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

def save_as_mp3(audio_array: np.ndarray, output_path: str, sample_rate: int = 24000) -> str:
    """Convert numpy array to MP3 file and return the path"""
    wav_path = output_path.replace('.mp3', '_temp.wav')
    mp3_path = output_path
    
    # Save as WAV first
    sf.write(wav_path, audio_array, sample_rate)
    
    # Convert to MP3
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format='mp3')
    
    # Clean up temporary WAV file
    os.remove(wav_path)
    return mp3_path

@app.post("/tts/")
async def synthesize(req: TTSRequest):
    # 1) Validate
    lang = req.language
    gen  = req.gender
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(400, f"Unsupported language '{lang}'. Choose from {sorted(SUPPORTED_LANGS)}.")
    if gen not in SUPPORTED_GENDERS:
        raise HTTPException(400, f"Unsupported gender '{gen}'. Choose 'male' or 'female'.")
    ref_wav = VOICE_REFS[lang][gen]
    if not os.path.isfile(ref_wav):
        raise HTTPException(500, f"Missing reference file: {ref_wav}")

    # 2) Chunk text per-language limit
    limits = {"hi": 250, "en": 300}
    max_len = limits.get(lang, 300)
    chunks = chunk_text(req.text, lang, max_len)
    if not chunks:
        raise HTTPException(500, "Text chunking failed.")

    # 3) Synthesize each chunk to a numpy waveform
    waves: List[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        try:
            # Add slight overlap in chunks by including a few words from next chunk
            if i < len(chunks) - 1:
                words = chunks[i+1].split()
                overlap_text = ' '.join(words[:2]) if words else ''
                chunk_with_overlap = f"{chunk} {overlap_text}"
            else:
                chunk_with_overlap = chunk
                
            wav = tts.tts(text=chunk_with_overlap, speaker_wav=ref_wav, language=lang)
            wav = normalize_audio(wav)
            wav = apply_lowpass(wav)
            waves.append(wav)
        except Exception as e:
            raise HTTPException(500, f"TTS generation failed on chunk '{chunk}': {e}")

    # Concatenate with crossfading
    if not waves:
        raise HTTPException(500, "No audio generated")
        
    result = waves[0]
    for wav in waves[1:]:
        result = crossfade(result, wav)
    
    # Write to disk
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{lang}_{gen}_{ts}.mp3"
    out_path = os.path.join(out_dir, filename)
    out_path = save_as_mp3(result, out_path)

    return FileResponse(out_path, media_type="audio/mp3", filename=filename)

@app.post("/clone-voice")
async def clone_voice(
    text: str = Form(...),
    language: str = Form(default="hi"),
    reference_audio: UploadFile = File(...)
):
    try:
        unique_id = str(uuid.uuid4())
        ref_audio_path = f"temp_ref_{unique_id}.wav"
        output_dir = "output"  # changed from "tts_outputs" to "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cloned_voice_{unique_id}.mp3")
        
        # Save uploaded reference audio
        with open(ref_audio_path, "wb") as f:
            content = await reference_audio.read()
            f.write(content)
        
        # Chunk text if too long
        limits = {"hi": 250, "en": 300}
        max_len = limits.get(language, 300)
        text_chunks = chunk_text(text, language, max_len)
        audio_chunks = []
        
        # Generate audio for each chunk
        for i, chunk in enumerate(text_chunks):
            # Add overlap with next chunk
            if i < len(text_chunks) - 1:
                words = text_chunks[i+1].split()
                overlap_text = ' '.join(words[:2]) if words else ''
                chunk_with_overlap = f"{chunk} {overlap_text}"
            else:
                chunk_with_overlap = chunk
                
            wav = tts.tts(text=chunk_with_overlap, language=language, speaker_wav=ref_audio_path)
            wav = normalize_audio(wav)
            wav = apply_lowpass(wav)
            audio_chunks.append(wav)
        
        # Combine chunks with crossfading
        if audio_chunks:
            result = audio_chunks[0]
            for chunk in audio_chunks[1:]:
                result = crossfade(result, chunk)
        
        # Save combined audio as MP3
        output_path = save_as_mp3(result, output_path)
        
        # Clean up temp reference file
        os.remove(ref_audio_path)
        
        return FileResponse(
            path=output_path,
            media_type="audio/mp3",
            filename=f"cloned_voice_{unique_id}.mp3"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "4-Way Multilingual TTS API is up and running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
