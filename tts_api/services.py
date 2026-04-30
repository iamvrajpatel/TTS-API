import asyncio
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tts_api.catalog import LanguageCatalog


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_MODEL_ROOT = BASE_DIR / "models"


def _repo_dir(repo_id: str) -> Path:
    return LOCAL_MODEL_ROOT / repo_id.replace("/", "--")


class ModelNotReadyError(RuntimeError):
    """Raised when synthesis is requested before the model is loaded."""


class ServiceBusyError(RuntimeError):
    """Raised when the model is already handling another request."""


class ModelLoadError(RuntimeError):
    """Raised when model startup failed due to dependency or runtime issues."""


@dataclass
class LoadedModelBundle:
    model: Any
    tokenizer: Any
    description_tokenizer: Any
    device: str
    sampling_rate: int
    model_name: str


def build_default_model_loader(
    model_name: str = "ai4bharat/indic-parler-tts",
    model_root: Path = LOCAL_MODEL_ROOT,
) -> Callable[[], LoadedModelBundle]:
    def load() -> LoadedModelBundle:
        from huggingface_hub import snapshot_download
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        model_root.mkdir(parents=True, exist_ok=True)
        model_dir = _repo_dir(model_name)
        snapshot_download(repo_id=model_name, local_dir=str(model_dir))

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = ParlerTTSForConditionalGeneration.from_pretrained(str(model_dir)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        text_encoder_name = model.config.text_encoder._name_or_path
        text_encoder_dir = _repo_dir(text_encoder_name)
        snapshot_download(repo_id=text_encoder_name, local_dir=str(text_encoder_dir))
        description_tokenizer = AutoTokenizer.from_pretrained(str(text_encoder_dir))
        return LoadedModelBundle(
            model=model,
            tokenizer=tokenizer,
            description_tokenizer=description_tokenizer,
            device=device,
            sampling_rate=model.config.sampling_rate,
            model_name=model_name,
        )

    return load


class IndicParlerTTSService:
    def __init__(
        self,
        catalog: LanguageCatalog,
        model_loader: Callable[[], LoadedModelBundle] | None = None,
    ) -> None:
        self._catalog = catalog
        self._model_loader = model_loader or build_default_model_loader()
        self._bundle: LoadedModelBundle | None = None
        self._load_error: str | None = None
        self._load_lock = asyncio.Lock()
        self._synthesis_semaphore = asyncio.Semaphore(1)

    @property
    def is_ready(self) -> bool:
        return self._bundle is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str | None:
        return self._bundle.device if self._bundle else None

    @property
    def model_name(self) -> str:
        return self._bundle.model_name if self._bundle else "ai4bharat/indic-parler-tts"

    async def load(self) -> None:
        if self._bundle is not None or self._load_error is not None:
            return

        async with self._load_lock:
            if self._bundle is not None or self._load_error is not None:
                return

            try:
                self._bundle = self._model_loader()
            except Exception as exc:
                self._load_error = self._build_load_error_message(exc)
                logger.exception("Failed to load Indic Parler TTS model")

    def _build_load_error_message(self, exc: Exception) -> str:
        base_message = f"Model loading failed: {exc}"
        raw_message = str(exc)

        if "libcudart.so.13" in raw_message:
            return (
                f"{base_message}. Detected a Torch/Torchaudio CUDA mismatch. "
                "Reinstall matching wheels for torch==2.7.1 and torchaudio==2.7.1, "
                "using either the CPU index or the cu126 index."
            )

        return base_message

    async def synthesize(
        self,
        language_code: str,
        text: str,
        speaker_name: str | None = None,
        voice_description: str | None = None,
    ) -> bytes:
        if self._load_error is not None:
            raise ModelLoadError(self._load_error)

        if self._bundle is None:
            raise ModelNotReadyError("The TTS model is still loading.")

        description = self._resolve_description(
            language_code=language_code,
            speaker_name=speaker_name,
            voice_description=voice_description,
        )

        try:
            await asyncio.wait_for(self._synthesis_semaphore.acquire(), timeout=0.01)
        except asyncio.TimeoutError as exc:
            raise ServiceBusyError("The server is currently processing another request.") from exc

        try:
            return self._synthesize_sync(
                self._bundle,
                text,
                description,
            )
        finally:
            self._synthesis_semaphore.release()

    def _resolve_description(
        self,
        language_code: str,
        speaker_name: str | None,
        voice_description: str | None,
    ) -> str:
        if voice_description:
            self._catalog.get_language(language_code)
            return voice_description

        if not speaker_name:
            raise ValueError("Either speaker_name or voice_description must be provided.")

        speaker = self._catalog.get_speaker(language_code, speaker_name)
        return speaker.description

    def _synthesize_sync(
        self,
        bundle: LoadedModelBundle,
        text: str,
        description: str,
    ) -> bytes:
        description_inputs = bundle.description_tokenizer(
            description, return_tensors="pt"
        ).to(bundle.device)
        prompt_inputs = bundle.tokenizer(text, return_tensors="pt").to(bundle.device)

        generation = bundle.model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

        audio_array = self._coerce_audio_array(generation)
        return self._encode_wav(audio_array, bundle.sampling_rate)

    def _coerce_audio_array(self, generation: Any) -> np.ndarray:
        if hasattr(generation, "cpu"):
            generation = generation.cpu()
        if hasattr(generation, "numpy"):
            generation = generation.numpy()

        audio_array = np.asarray(generation, dtype=np.float32).squeeze()
        if audio_array.ndim != 1:
            audio_array = audio_array.reshape(-1)
        return audio_array

    def _encode_wav(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        return buffer.getvalue()
