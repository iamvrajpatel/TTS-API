from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from tts_api.catalog import (
    LanguageCatalog,
    UnsupportedLanguageError,
    UnsupportedSpeakerError,
    build_default_catalog,
)
from tts_api.schemas import TTSRequest
from tts_api.services import (
    IndicParlerTTSService,
    ModelLoadError,
    ModelNotReadyError,
    ServiceBusyError,
)


BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
CUSTOM_DESCRIPTION_PLACEHOLDER = (
    "A warm, expressive female voice with a slightly brisk pace, "
    "clear pronunciation, and a clean studio recording with almost no background noise."
)


def _build_filename(language_code: str, voice_label: str) -> str:
    safe_label = voice_label.strip().replace(" ", "_").lower()
    return f"{language_code}-{safe_label}.wav"


def create_app(
    service: IndicParlerTTSService | None = None,
    catalog: LanguageCatalog | None = None,
) -> FastAPI:
    language_catalog = catalog or build_default_catalog()
    tts_service = service or IndicParlerTTSService(catalog=language_catalog)

    app = FastAPI(title="AI4Bharat Indic Parler TTS API", version="1.0.0")

    @app.on_event("startup")
    async def startup() -> None:
        await tts_service.load()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "languages": language_catalog.as_template_data(),
                "custom_description_placeholder": CUSTOM_DESCRIPTION_PLACEHOLDER,
            },
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        ready = tts_service.is_ready
        load_error = tts_service.load_error
        if ready:
            status = "ok"
            status_code = 200
        elif load_error:
            status = "error"
            status_code = 503
        else:
            status = "loading"
            status_code = 503

        return JSONResponse(
            {
                "status": status,
                "ready": ready,
                "model_loaded": ready,
                "model_name": tts_service.model_name,
                "device": tts_service.device,
                "model_error": load_error,
            },
            status_code=status_code,
        )

    @app.post("/tts/")
    async def synthesize(request_body: TTSRequest) -> Response:
        voice_label = request_body.speaker or "custom-description"
        try:
            wav_bytes = await tts_service.synthesize(
                language_code=request_body.language,
                text=request_body.text,
                speaker_name=request_body.speaker,
                voice_description=request_body.voice_description,
            )
        except UnsupportedLanguageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except UnsupportedSpeakerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelLoadError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ServiceBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Unexpected error during speech generation.",
            ) from exc

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="{_build_filename(request_body.language, voice_label)}"'
            },
        )

    return app
