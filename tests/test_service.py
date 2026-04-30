import asyncio
import unittest

import numpy as np

from tts_api.catalog import build_default_catalog
from tts_api.services import (
    IndicParlerTTSService,
    LoadedModelBundle,
    ModelLoadError,
    ModelNotReadyError,
    ServiceBusyError,
)


class FakeBatch:
    def __init__(self) -> None:
        self.input_ids = [[1, 2, 3]]
        self.attention_mask = [[1, 1, 1]]

    def to(self, device: str) -> "FakeBatch":
        self.device = device
        return self


class FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> FakeBatch:
        self.last_text = text
        self.last_return_tensors = return_tensors
        return FakeBatch()


class FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = np.array(values, dtype=np.float32)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_calls.append(kwargs)
        return FakeTensor([[0.1, -0.1, 0.2, -0.2]])


def fake_loader() -> LoadedModelBundle:
    return LoadedModelBundle(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        description_tokenizer=FakeTokenizer(),
        device="cpu",
        sampling_rate=24000,
        model_name="fake-model",
    )


class BusySemaphore:
    async def acquire(self) -> None:
        await asyncio.sleep(1)

    def release(self) -> None:
        return None


class ServiceTests(unittest.TestCase):
    def create_service(self) -> IndicParlerTTSService:
        return IndicParlerTTSService(
            catalog=build_default_catalog(),
            model_loader=fake_loader,
        )

    def test_startup_preload_succeeds(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            await service.load()
            self.assertTrue(service.is_ready)
            self.assertEqual(service.model_name, "fake-model")

        asyncio.run(scenario())

    def test_model_not_ready_raises(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            with self.assertRaises(ModelNotReadyError):
                await service.synthesize("hi", "Divya", "Namaste")

        asyncio.run(scenario())

    def test_synthesis_uses_speaker_description(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            await service.load()
            service._encode_wav = lambda audio, sample_rate: b"WAV"  # type: ignore[method-assign]

            wav_bytes = await service.synthesize(
                language_code="hi",
                speaker_name="Divya",
                text="Namaste",
            )

            self.assertEqual(wav_bytes, b"WAV")
            self.assertEqual(service._bundle.description_tokenizer.last_return_tensors, "pt")
            self.assertIn(
                "Divya's voice sounds",
                service._bundle.description_tokenizer.last_text,
            )

        asyncio.run(scenario())

    def test_concurrency_guard_rejects_parallel_generation(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            await service.load()
            service._synthesis_semaphore = BusySemaphore()  # type: ignore[assignment]

            with self.assertRaises(ServiceBusyError):
                await service.synthesize(
                    language_code="hi",
                    speaker_name="Divya",
                    text="Second request",
                )

        asyncio.run(scenario())

    def test_loader_failure_is_retained_as_model_error(self) -> None:
        async def scenario() -> None:
            def broken_loader() -> LoadedModelBundle:
                raise OSError("libcudart.so.13: cannot open shared object file")

            service = IndicParlerTTSService(
                catalog=build_default_catalog(),
                model_loader=broken_loader,
            )
            await service.load()

            self.assertFalse(service.is_ready)
            self.assertIsNotNone(service.load_error)
            self.assertIn("torchaudio", service.load_error.lower())

            with self.assertRaises(ModelLoadError):
                await service.synthesize(
                    language_code="hi",
                    speaker_name="Divya",
                    text="Namaste",
                )

        asyncio.run(scenario())

    def test_custom_description_bypasses_catalog_speaker_lookup(self) -> None:
        async def scenario() -> None:
            service = self.create_service()
            await service.load()
            service._encode_wav = lambda audio, sample_rate: b"WAV"  # type: ignore[method-assign]

            wav_bytes = await service.synthesize(
                language_code="hi",
                text="Namaste",
                voice_description="custom calm voice with clean studio sound",
            )

            self.assertEqual(wav_bytes, b"WAV")
            self.assertEqual(
                service._bundle.description_tokenizer.last_text,
                "custom calm voice with clean studio sound",
            )

        asyncio.run(scenario())
