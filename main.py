import os
import uuid
import logging
import threading
from datetime import datetime
import numpy as np
import torch
import asyncio
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, constr
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from transformers import GPT2Model
from transformers.generation.utils import GenerationMixin

from utils.utils import chunk_text, create_silence_padding, run_tts_task, \
    synthesize_tts_chunks, clone_voice_chunks
from utils.others import save_as_mp3
from utils.remove_bg import clean_and_extend_audio

# Create a global threading lock
processing_lock = threading.Lock()

# Logging
logger = logging.getLogger("uvicorn.error")

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

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load on startup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tts: TTS

@app.on_event("startup")
async def load_model():
    global tts
    logger.info("Loading TTS model...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
              progress_bar=False).to(device)
    logger.info("Model loaded successfully.")

# Reference voices
VOICE_REFS = {
    "en": {"male": "refs/en_male_ref.mp3", "female": "refs/en_female_ref.mp3"},
    "hi": {"male": "refs/hi_male_ref.mp3", "female": "refs/hi_female_ref.mp3"},
}
SUPPORTED_LANGS = set(VOICE_REFS.keys())
SUPPORTED_GENDERS = {"male", "female"}

class TTSRequest(BaseModel):
    text: constr(min_length=1)
    language: constr(to_lower=True)
    gender: constr(to_lower=True)

# ------------------- SUPPORT FUNCTIONS -------------------

MAX_WORKERS = 1
tts_semaphore = asyncio.Semaphore(MAX_WORKERS)


# Helper wrapper for TTS
async def run_tts_task(func, *args, **kwargs):
    async with tts_semaphore:
        return await asyncio.get_event_loop().run_in_executor(None, func, *args, *kwargs)

def synthesize_tts_chunks(chunks, ref_wav, lang, request):
    waves = []
    for chunk in chunks:
        wav = tts.tts(text=chunk, speaker_wav=ref_wav, language=lang)
        waves.append(wav)
    return waves

def clone_voice_chunks(chunks, ref_wav, lang, request):
    waves = []
    for chunk in chunks:
        wav = tts.tts(text=chunk, speaker_wav=ref_wav, language=lang)
        waves.append(wav)
    return waves

# ------------------- ROUTES -------------------

@app.post("/tts/")
async def synthesize(req: TTSRequest, request: Request):
    acquired = processing_lock.acquire(blocking=False)
    
    if not acquired:
        raise HTTPException(status_code=503, detail=f"Server is busy! Try after sometime!!")
    
    temp_files = []
    try:
        
        lang = req.language
        gen = req.gender
        if lang not in SUPPORTED_LANGS:
            raise HTTPException(400, f"Unsupported language '{lang}'.")
        if gen not in SUPPORTED_GENDERS:
            raise HTTPException(400, f"Unsupported gender '{gen}'.")
        ref_wav = VOICE_REFS[lang][gen]
        if not os.path.isfile(ref_wav):
            raise HTTPException(500, f"Missing reference file: {ref_wav}")

        # Chunk text
        limits = {"hi": 230, "en": 280}
        chunks = chunk_text(req.text, lang, limits.get(lang, 300))
        if not chunks:
            raise HTTPException(500, "Text chunking failed.")

        # Run synthesis
        waves = await run_tts_task(synthesize_tts_chunks, chunks, ref_wav, lang, request)
        if not waves:
            raise HTTPException(500, "No audio generated.")

        # Concatenate
        silence = create_silence_padding(sample_rate=24000, duration_ms=5)
        result = waves[0]
        for wav in waves[1:]:
            result = np.concatenate([result, silence, wav])

        # Save
        os.makedirs("output", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{lang}_{gen}_{ts}.mp3"
        out_path = os.path.join("output", filename)
        save_as_mp3(result, out_path)

        return FileResponse(out_path, media_type="audio/mp3", filename=filename)
    
    except Exception as e:
        logger.exception("Error in /tts/")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        processing_lock.release()
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

@app.post("/clone-voice")
async def clone_voice(
    request: Request,
    text: str = Form(...),
    language: str = Form(default="hi"),
    reference_audio: UploadFile = File(...)
):
    
    acquired = processing_lock.acquire(blocking=False)
    
    if not acquired:
        raise HTTPException(status_code=503, detail=f"Server is busy! Try after sometime!!")
    
    temp_files = []
    try:
        unique_id = str(uuid.uuid4())
        raw_ref_audio_path = f"temp_ref_{unique_id}_raw.wav"
        cleaned_ref_audio_path = f"temp_ref_{unique_id}_cleaned.wav"
        temp_files.extend([raw_ref_audio_path, cleaned_ref_audio_path])
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cloned_voice_{unique_id}.mp3")

        # Save uploaded audio
        content = await reference_audio.read()
        if not content:
            raise HTTPException(400, "Empty reference audio file.")
        with open(raw_ref_audio_path, "wb") as f:
            f.write(content)

        logger.info(f"Uploaded file: {reference_audio.filename}, size={len(content)} bytes")

        # Clean reference audio (short duration for testing)
        ref_out_cleaned = clean_and_extend_audio(
            raw_ref_audio_path, cleaned_ref_audio_path, min_duration_sec=5
        )

        # Chunk text
        limits = {"hi": 230, "en": 280}
        chunks = chunk_text(text, language, limits.get(language, 290))
        if not chunks:
            raise HTTPException(500, "Text chunking failed.")

        # Run synthesis
        audio_chunks = await run_tts_task(clone_voice_chunks, chunks, ref_out_cleaned, language, request)
        if not audio_chunks:
            raise HTTPException(500, "No audio generated.")

        # Concatenate
        silence = create_silence_padding(sample_rate=24000, duration_ms=5)
        result = audio_chunks[0]
        for chunk in audio_chunks[1:]:
            result = np.concatenate([result, silence, chunk])

        # Save
        save_as_mp3(result, output_path)

        return FileResponse(output_path, media_type="audio/mp3", filename=f"cloned_voice_{unique_id}.mp3")
    
    except Exception as e:
        logger.exception("Error in /clone-voice")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        processing_lock.release()
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

@app.get("/text-to-speech", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/")
async def root():
    return {"message": "4-Way Multilingual TTS API is up and running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
