from tts_api.catalog import build_default_catalog
from tts_api.services import (
    IndicParlerTTSService,
    SynthesisConcurrencyGate,
    XttsVoiceCloneService,
)
from tts_api.web import create_app


catalog = build_default_catalog()
generation_gate = SynthesisConcurrencyGate(max_concurrent_requests=1)
tts_service = IndicParlerTTSService(catalog=catalog, generation_gate=generation_gate)
voice_clone_service = XttsVoiceCloneService(
    catalog=catalog,
    generation_gate=generation_gate,
)
app = create_app(
    service=tts_service,
    clone_service=voice_clone_service,
    catalog=catalog,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
