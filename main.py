# main.py
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, constr
from TTS.api import TTS

app = FastAPI(title="4-Way Multilingual TTS (Hindi/English x Male/Female)")

# Load the multilingual XTTS-v2 model (17 languages incl. en & hi) :contentReference[oaicite:0]{index=0}
MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(MODEL_ID)

# Map each (language, gender) to its own 6-sec reference WAV for voice cloning :contentReference[oaicite:1]{index=1}
VOICE_REFS = {
    "en": {
        "male":   "refs/en_male_ref.mp3",
        "female": "refs/en_female_ref.mp3",
    },
    "hi": {
        "male":   "refs/hi_male_ref.mp3",
        "female": "refs/hi_female_ref.mp3",
    },
}

# Only allow these two ISO codes
SUPPORTED_LANGS = set(VOICE_REFS.keys())
SUPPORTED_GENDERS = {"male", "female"}

class TTSRequest(BaseModel):
    text:     constr(min_length=1)
    language: constr(to_lower=True)
    gender:   constr(to_lower=True)

@app.post("/tts/")
async def synthesize(req: TTSRequest):
    # 1) Validate language
    if req.language not in SUPPORTED_LANGS:
        raise HTTPException(400, f"Unsupported language '{req.language}'. Choose from {sorted(SUPPORTED_LANGS)}.")

    # 2) Validate gender
    if req.gender not in SUPPORTED_GENDERS:
        raise HTTPException(400, f"Unsupported gender '{req.gender}'. Choose 'male' or 'female'.")

    # 3) Lookup and verify reference file
    ref_wav = VOICE_REFS[req.language][req.gender]
    if not os.path.isfile(ref_wav):
        raise HTTPException(
            500,
            f"Missing reference file for {req.language}/{req.gender}: {ref_wav}. "
            "Make sure the file exists."
        )

    # 4) Prepare output filename
    out_dir = "tts_outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{req.language}_{req.gender}_{ts}.wav"
    out_path = os.path.join(out_dir, filename)

    # 5) Synthesize with voice-cloning API
    try:
        tts.tts_to_file(
            text=req.text,
            speaker_wav=[ref_wav],
            language=req.language,
            file_path=out_path
        )
    except Exception as e:
        raise HTTPException(500, f"TTS synthesis failed: {e}")

    # 6) Return the WAV file
    return FileResponse(out_path, media_type="audio/wav", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
