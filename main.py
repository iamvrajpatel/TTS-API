import os
import uuid
from datetime import datetime
from typing import List, Tuple
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, constr
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from transformers import GPT2Model
from transformers.generation.utils import GenerationMixin

from utils.utils import chunk_text, apply_lowpass, normalize_audio, crossfade
from utils.others import save_as_mp3
from utils.remove_bg import clean_and_extend_audio

import asyncio


# Make GPT2Model compatible with GenerationMixin
if not issubclass(GPT2Model, GenerationMixin):
    GPT2Model.__bases__ += (GenerationMixin,)
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs,
])

app = FastAPI(title="4-Way Multilingual TTS (Hindi/English x Male/Female) with Chunking")

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

MAX_WORKERS = 1  # Set your desired concurrency limit here
tts_semaphore = asyncio.Semaphore(MAX_WORKERS)

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
        # Check for cancellation
        if await request.is_disconnected():
            raise asyncio.CancelledError()
        try:
            # Add slight overlap in chunks by including a few words from next chunk
            if i < len(chunks) - 1:
                words = chunks[i+1].split()
                overlap_text = ' '.join(words[:2]) if words else ''
                chunk_with_overlap = f"{chunk} {overlap_text}"
            else:
                chunk_with_overlap = chunk
            wav = await asyncio.to_thread(tts.tts, text=chunk_with_overlap, speaker_wav=ref_wav, language=lang)
            wav = normalize_audio(wav)
            wav = apply_lowpass(wav)
            waves.append(wav)
        except Exception as e:
            raise HTTPException(500, f"TTS generation failed on chunk '{chunk}': {e}")
    return waves

@app.post("/tts/")
async def synthesize(req: TTSRequest, request: Request):
    temp_files = []
    try:
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
        limits = {"hi": 230, "en": 280}
        max_len = limits.get(lang, 300)
        chunks = chunk_text(req.text, lang, max_len)
        if not chunks:
            raise HTTPException(500, "Text chunking failed.")

        # Queue and run TTS synthesis
        waves = await run_tts_task(synthesize_tts_chunks, chunks, ref_wav, lang, request)

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
    finally:
        # Remove temp files if any
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

async def clone_voice_chunks(text_chunks, cleaned_ref_audio_path, language, request: Request):
    audio_chunks = []
    for i, chunk in enumerate(text_chunks):
        # Check for cancellation
        if await request.is_disconnected():
            raise asyncio.CancelledError()
        try:
            # Add overlap with next chunk
            if i < len(text_chunks) - 1:
                words = text_chunks[i+1].split()
                overlap_text = ' '.join(words[:2]) if words else ''
                chunk_with_overlap = f"{chunk} {overlap_text}"
            else:
                chunk_with_overlap = chunk
            wav = await asyncio.to_thread(tts.tts, text=chunk_with_overlap, language=language, speaker_wav=cleaned_ref_audio_path)
            wav = normalize_audio(wav)
            wav = apply_lowpass(wav)
            audio_chunks.append(wav)
        except Exception as e:
            raise HTTPException(500, f"TTS generation failed on chunk '{chunk}': {e}")
    return audio_chunks

@app.post("/clone-voice")
async def clone_voice(
    request: Request,
    text: str = Form(...),
    language: str = Form(default="hi"),
    reference_audio: UploadFile = File(...)
):
    temp_files = []
    try:
        unique_id = str(uuid.uuid4())
        raw_ref_audio_path = f"temp_ref_{unique_id}_raw.wav"
        cleaned_ref_audio_path = f"temp_ref_{unique_id}_cleaned.wav"
        temp_files.extend([raw_ref_audio_path, cleaned_ref_audio_path])
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cloned_voice_{unique_id}.mp3")

        # Save uploaded reference audio
        with open(raw_ref_audio_path, "wb") as f:
            content = await reference_audio.read()
            f.write(content)

        # Clean and extend reference audio
        ref_out_cleaned = clean_and_extend_audio(raw_ref_audio_path, cleaned_ref_audio_path, min_duration_sec=120)

        # Chunk text if too long
        limits = {"hi": 230, "en": 280}
        max_len = limits.get(language, 290)
        text_chunks = chunk_text(text, language, max_len)

        # Queue and run TTS synthesis
        audio_chunks = await run_tts_task(clone_voice_chunks, text_chunks, ref_out_cleaned, language, request)

        # Combine chunks with crossfading
        if audio_chunks:
            result = audio_chunks[0]
            for chunk in audio_chunks[1:]:
                result = crossfade(result, chunk)

        # Save combined audio as MP3
        output_path = save_as_mp3(result, output_path)

        return FileResponse(
            path=output_path,
            media_type="audio/mp3",
            filename=f"cloned_voice_{unique_id}.mp3"
        )
    except asyncio.CancelledError:
        raise HTTPException(status_code=499, detail="Request cancelled by client.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove temp files if any
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

@app.get("/")
async def root():
    return {"message": "4-Way Multilingual TTS API is up and running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
