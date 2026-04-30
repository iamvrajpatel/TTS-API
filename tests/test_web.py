import unittest

from tts_api.catalog import build_default_catalog
from tts_api.services import ModelLoadError
from tts_api.services import ModelNotReadyError

try:
    from fastapi.testclient import TestClient
    from tts_api.web import create_app

    FASTAPI_TESTS_AVAILABLE = True
except ModuleNotFoundError:
    TestClient = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]
    FASTAPI_TESTS_AVAILABLE = False


class FakeWebService:
    def __init__(
        self,
        catalog,
        ready: bool = True,
        load_error: str | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        self.catalog = catalog
        self.ready = ready
        self._load_error = load_error
        self.calls: list[tuple[str, str, str]] = []

    @property
    def is_ready(self) -> bool:
        return self.ready

    @property
    def is_generating(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "fake-web-model"

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str | None:
        return "cpu" if self.ready else None

    async def load(self) -> None:
        return None

    async def synthesize(
        self,
        language_code: str,
        text: str,
        speaker_name: str | None = None,
        voice_description: str | None = None,
    ) -> bytes:
        if self._load_error:
            raise ModelLoadError(self._load_error)
        if not self.ready:
            raise ModelNotReadyError("The TTS model is still loading.")
        if voice_description:
            self.catalog.get_language(language_code)
        else:
            self.catalog.get_speaker(language_code, speaker_name)
        self.calls.append((language_code, speaker_name or "custom-description", text))
        return b"RIFFfakewav"


@unittest.skipUnless(FASTAPI_TESTS_AVAILABLE, "fastapi is not installed")
class WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = build_default_catalog()

    def test_root_renders_tts_page(self) -> None:
        with TestClient(
            create_app(service=FakeWebService(self.catalog), catalog=self.catalog)
        ) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Backend sync", response.text)

    def test_health_reflects_loaded_state(self) -> None:
        with TestClient(
            create_app(
                service=FakeWebService(self.catalog, ready=True),
                catalog=self.catalog,
            )
        ) as ready_client:
            ready_response = ready_client.get("/health")

        with TestClient(
            create_app(
                service=FakeWebService(self.catalog, ready=False),
                catalog=self.catalog,
            )
        ) as loading_client:
            loading_response = loading_client.get("/health")

        self.assertEqual(ready_response.status_code, 200)
        self.assertTrue(ready_response.json()["ready"])
        self.assertFalse(ready_response.json()["generation_in_progress"])
        self.assertEqual(loading_response.status_code, 503)
        self.assertFalse(loading_response.json()["ready"])

    def test_health_reports_model_error(self) -> None:
        with TestClient(
            create_app(
                service=FakeWebService(
                    self.catalog,
                    ready=False,
                    load_error="Model loading failed: torchaudio mismatch",
                ),
                catalog=self.catalog,
            )
        ) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "error")
        self.assertIn("torchaudio mismatch", response.json()["model_error"])

    def test_tts_returns_audio_for_valid_request(self) -> None:
        service = FakeWebService(self.catalog, ready=True)
        with TestClient(create_app(service=service, catalog=self.catalog)) as client:
            response = client.post(
                "/tts/",
                json={
                    "text": "Namaste",
                    "language": "hi",
                    "voice_mode": "speaker",
                    "speaker": "Divya",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "audio/wav")
            self.assertIn("X-Generation-Time-Ms", response.headers)
            self.assertEqual(service.calls, [("hi", "Divya", "Namaste")])

    def test_tts_rejects_invalid_speaker(self) -> None:
        with TestClient(
            create_app(service=FakeWebService(self.catalog), catalog=self.catalog)
        ) as client:
            response = client.post(
                "/tts/",
                json={
                    "text": "Namaste",
                    "language": "hi",
                    "voice_mode": "speaker",
                    "speaker": "Mary",
                },
            )

            self.assertEqual(response.status_code, 400)

    def test_tts_rejects_empty_text(self) -> None:
        with TestClient(
            create_app(service=FakeWebService(self.catalog), catalog=self.catalog)
        ) as client:
            response = client.post(
                "/tts/",
                json={
                    "text": "   ",
                    "language": "hi",
                    "voice_mode": "speaker",
                    "speaker": "Divya",
                },
            )

            self.assertEqual(response.status_code, 422)

    def test_tts_accepts_custom_description_mode(self) -> None:
        service = FakeWebService(self.catalog, ready=True)
        with TestClient(create_app(service=service, catalog=self.catalog)) as client:
            response = client.post(
                "/tts/",
                json={
                    "text": "Namaste",
                    "language": "hi",
                    "voice_mode": "description",
                    "voice_description": "warm custom voice with a calm pace",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "audio/wav")
            self.assertEqual(
                service.calls,
                [("hi", "custom-description", "Namaste")],
            )
