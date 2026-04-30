# AI4Bharat Indic Parler TTS API

FastAPI app for hosted text-to-speech using `ai4bharat/indic-parler-tts`, with an HTML interface and explicit speaker selection for multiple Indian languages.

## Requirements

- Python 3.11 is recommended
- CUDA-enabled PyTorch is optional; the app falls back to CPU when CUDA is unavailable

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are fixing an existing broken Torch or Torchaudio install, reinstall matching wheels first.

CPU-only:

```bash
pip uninstall -y torch torchaudio
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchaudio==2.7.1
pip install -r requirements.txt
```

CUDA 12.6:

```bash
pip uninstall -y torch torchaudio
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchaudio==2.7.1
pip install -r requirements.txt
```

## Run the app

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/` for the hosted HTML interface.

## Model storage

The app stores Hugging Face model files inside this project instead of the global cache.

Default local model path:

```text
TTS-API/models/ai4bharat--indic-parler-tts
```

The text encoder tokenizer is also stored locally under:

```text
TTS-API/models/<text-encoder-repo-name-with-slashes-replaced-by-->
```

These folders are created automatically on first startup or first download.

## Troubleshooting

If startup fails with `libcudart.so.13` or another `torchaudio` shared-library error, the environment has incompatible Torch and Torchaudio wheels.

Expected versions for this repo:

- `torch==2.7.1`
- `torchaudio==2.7.1`

After reinstalling, confirm with:

```bash
python - <<'PY'
import torch
import torchaudio
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
PY
```

## API

### `GET /`

Serves the TTS HTML page.

### `GET /health`

Returns readiness and model-load details.

### `POST /tts/`

Generates a WAV response from JSON input.

Using an existing speaker:

```json
{
  "text": "अरे, तुम आज कैसे हो?",
  "language": "hi",
  "voice_mode": "speaker",
  "speaker": "Divya"
}
```

Using a custom description:

```json
{
  "text": "अरे, तुम आज कैसे हो?",
  "language": "hi",
  "voice_mode": "description",
  "voice_description": "A warm, expressive female voice with a slightly brisk pace, clear pronunciation, and a clean studio recording with almost no background noise."
}
```

Example with `curl`:

```bash
curl --request POST "http://localhost:8000/tts/" \
  --header "Content-Type: application/json" \
  --output sample.wav \
  --data '{
    "text": "Namaste from the hosted Indic Parler TTS API.",
    "language": "en",
    "speaker": "Mary"
  }'
```

## Supported languages

Assamese, Bengali, Bodo, Chhattisgarhi, Dogri, English, Gujarati, Hindi, Kannada, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu.

The UI is driven by the same server-side language catalog used for API validation, including recommended speakers for each language.
