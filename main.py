from tts_api.catalog import build_default_catalog
from tts_api.services import IndicParlerTTSService
from tts_api.web import create_app


catalog = build_default_catalog()
tts_service = IndicParlerTTSService(catalog=catalog)
app = create_app(service=tts_service, catalog=catalog)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
