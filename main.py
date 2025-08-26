import os
import uuid
from datetime import datetime
from typing import List, Tuple

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
    for chunk in chunks:
        try:
            wav = tts.tts(text=chunk, speaker_wav=ref_wav, language=lang)
        except Exception as e:
            raise HTTPException(500, f"TTS generation failed on chunk '{chunk}': {e}")
        waves.append(wav)

    # 4) Concatenate and write to disk
    combined = np.concatenate(waves, axis=0)
    out_dir = "output"  # changed from "tts_outputs" to "output"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{lang}_{gen}_{ts}.wav"
    out_path = os.path.join(out_dir, filename)
    # assuming model outputs at 24000 Hz
    sf.write(out_path, combined, 24000)

    return FileResponse(out_path, media_type="audio/wav", filename=filename)

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
        output_path = os.path.join(output_dir, f"cloned_voice_{unique_id}.wav")
        
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
        for chunk in text_chunks:
            wav = tts.tts(text=chunk, language=language, speaker_wav=ref_audio_path)
            audio_chunks.append(wav)
        
        # Combine audio chunks
        combined_audio = np.concatenate(audio_chunks)
        
        # Save combined audio
        sf.write(output_path, combined_audio, 24000)
        
        # Clean up temp reference file
        os.remove(ref_audio_path)
        
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=f"cloned_voice_{unique_id}.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "4-Way Multilingual TTS API is up and running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
