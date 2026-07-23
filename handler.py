from __future__ import annotations

import base64
import binascii
import http.client
import ipaddress
import logging
import math
import os
import random
import secrets
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import numpy as np
import runpod
import torch
from indextts.infer_v2 import IndexTTS2


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("indextts2-worker")

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "m4a", "ogg"}
EMOTION_VECTOR_SIZE = 8
MAX_REDIRECTS = 3
NORMALIZED_SAMPLE_RATE = 22_050


class RequestValidationError(ValueError):
    """A safe error that can be returned to the API caller."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/index-tts/checkpoints")).resolve()
CONFIG_PATH = Path(
    os.getenv("CONFIG_PATH", str(MODEL_DIR / "config.yaml"))
).resolve()
USE_FP16 = _env_bool("USE_FP16", True)
USE_DEEPSPEED = _env_bool("USE_DEEPSPEED", False)
USE_CUDA_KERNEL = _env_bool("USE_CUDA_KERNEL", False)
MAX_TEXT_LENGTH = _env_positive_int("MAX_TEXT_LENGTH", 5_000)
MAX_AUDIO_BYTES = _env_positive_int("MAX_AUDIO_BYTES", 25 * 1024 * 1024)
DOWNLOAD_TIMEOUT_SECONDS = _env_positive_int("DOWNLOAD_TIMEOUT_SECONDS", 30)


def _required_model_paths(model_dir: Path, config_path: Path) -> dict[str, Path]:
    cache = model_dir / "hf_cache"
    return {
        "config": config_path,
        "bpe": model_dir / "bpe.model",
        "gpt": model_dir / "gpt.pth",
        "s2mel": model_dir / "s2mel.pth",
        "wav2vec stats": model_dir / "wav2vec2bert_stats.pt",
        "speaker matrix": model_dir / "feat1.pt",
        "emotion matrix": model_dir / "feat2.pt",
        "emotion model": model_dir / "qwen0.6bemo4-merge" / "model.safetensors",
        "W2V-BERT config": cache / "w2v-bert-2.0" / "config.json",
        "W2V-BERT model": cache / "w2v-bert-2.0" / "model.safetensors",
        "W2V-BERT preprocessor": cache
        / "w2v-bert-2.0"
        / "preprocessor_config.json",
        "semantic codec": cache / "semantic_codec_model.safetensors",
        "CAMPPlus": cache / "campplus_cn_common.bin",
        "BigVGAN config": cache / "bigvgan" / "config.json",
        "BigVGAN model": cache / "bigvgan" / "bigvgan_generator.pt",
    }


def _load_model() -> IndexTTS2:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but no NVIDIA GPU is available")
    if USE_DEEPSPEED:
        try:
            import deepspeed  # noqa: F401
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "USE_DEEPSPEED is enabled but DeepSpeed is not installed correctly"
            ) from exc

    required = _required_model_paths(MODEL_DIR, CONFIG_PATH)
    missing = [label for label, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Required model files are missing: " + ", ".join(missing))

    gpu_name = torch.cuda.get_device_name(0)
    LOGGER.info("GPU: %s", gpu_name)
    LOGGER.info(
        "Initializing IndexTTS2 model_dir=%s config=%s fp16=%s deepspeed=%s "
        "cuda_kernel=%s",
        MODEL_DIR,
        CONFIG_PATH,
        USE_FP16,
        USE_DEEPSPEED,
        USE_CUDA_KERNEL,
    )

    cache = MODEL_DIR / "hf_cache"
    aux_paths = {
        "w2v_bert": str(cache / "w2v-bert-2.0"),
        "semantic_codec": str(cache / "semantic_codec_model.safetensors"),
        "campplus": str(cache / "campplus_cn_common.bin"),
        "bigvgan": str(cache / "bigvgan"),
    }
    try:
        model = IndexTTS2(
            cfg_path=str(CONFIG_PATH),
            model_dir=str(MODEL_DIR),
            use_fp16=USE_FP16,
            device="cuda:0",
            use_cuda_kernel=USE_CUDA_KERNEL,
            use_deepspeed=USE_DEEPSPEED,
            use_accel=False,
            use_torch_compile=False,
            aux_paths=aux_paths,
        )
    except Exception:
        LOGGER.exception("IndexTTS2 initialization failed")
        raise RuntimeError("IndexTTS2 initialization failed; check worker logs") from None
    if USE_CUDA_KERNEL and not model.use_cuda_kernel:
        raise RuntimeError(
            "USE_CUDA_KERNEL is enabled but the custom CUDA kernel did not load"
        )
    LOGGER.info("IndexTTS2 initialized")
    return model


MODEL = _load_model()
INFERENCE_LOCK = threading.Lock()


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RequestValidationError(f"{name} must be an object")
    return value


def _optional_bool(data: dict[str, Any], name: str, default: bool) -> bool:
    value = data.get(name, default)
    if not isinstance(value, bool):
        raise RequestValidationError(f"{name} must be a boolean")
    return value


def _optional_text(data: dict[str, Any], name: str) -> str | None:
    value = data.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_extension(value: Any, field: str) -> str:
    if value is None:
        return "wav"
    if not isinstance(value, str):
        raise RequestValidationError(f"{field} must be a string")
    extension = value.lower().removeprefix(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise RequestValidationError(
            f"{field} must be one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return extension


def _validate_public_ip(value: str) -> None:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError as exc:
        raise RequestValidationError("Audio URL resolved to an invalid address") from exc
    if not address.is_global:
        raise RequestValidationError("Audio URL must resolve only to public addresses")


def _validate_url(url: Any) -> tuple[str, str, int, list[str], str, str]:
    if not isinstance(url, str) or not url.strip():
        raise RequestValidationError("Audio URL must be a non-empty string")
    normalized_url = url.strip()
    parsed = urlsplit(normalized_url)
    if parsed.scheme.lower() != "https":
        raise RequestValidationError("Audio URL must use HTTPS")
    if not parsed.hostname or parsed.username or parsed.password:
        raise RequestValidationError("Audio URL is invalid")
    try:
        hostname = parsed.hostname.rstrip(".").lower().encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise RequestValidationError("Audio URL host is invalid") from exc
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise RequestValidationError("Audio URL host is not allowed")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise RequestValidationError("Audio URL port is invalid") from exc
    try:
        addresses = socket.getaddrinfo(
            hostname, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror as exc:
        raise RequestValidationError("Audio URL host could not be resolved") from exc
    if not addresses:
        raise RequestValidationError("Audio URL host could not be resolved")
    public_addresses = sorted(
        {entry[4][0] for entry in addresses},
        key=lambda value: (ipaddress.ip_address(value.split("%", 1)[0]).version, value),
    )
    for address in public_addresses:
        _validate_public_ip(address)
    target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    host_header = f"[{hostname}]" if ":" in hostname else hostname
    if port != 443:
        host_header = f"{host_header}:{port}"
    return normalized_url, hostname, port, public_addresses, target, host_header


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection whose TCP destination cannot be changed by DNS rebinding."""

    def __init__(
        self, hostname: str, port: int, addresses: list[str], timeout: int
    ) -> None:
        super().__init__(
            hostname,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._addresses = addresses

    def connect(self) -> None:
        last_error: OSError | None = None
        deadline = time.monotonic() + float(self.timeout)
        for address in self._addresses:
            raw_socket: socket.socket | None = None
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Connection timed out")
                raw_socket = socket.create_connection(
                    (address, self.port), remaining, self.source_address
                )
                self.sock = self._context.wrap_socket(
                    raw_socket, server_hostname=self.host
                )
                return
            except OSError as exc:
                last_error = exc
                if raw_socket is not None:
                    raw_socket.close()
        if last_error is not None:
            raise last_error
        raise OSError("No validated address is available")


def _download_audio(url: str, destination: Path) -> None:
    current_url = url
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
    for redirect_count in range(MAX_REDIRECTS + 1):
        (
            current_url,
            hostname,
            port,
            addresses,
            target,
            host_header,
        ) = _validate_url(current_url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestValidationError("Audio download timed out")
        connection = _PinnedHTTPSConnection(
            hostname, port, addresses, max(1, math.ceil(remaining))
        )
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Host": host_header,
                    "User-Agent": "IndexTTS2-Runpod-Worker/1.0",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                if redirect_count == MAX_REDIRECTS:
                    raise RequestValidationError("Audio URL has too many redirects")
                location = response.getheader("Location")
                if not location:
                    raise RequestValidationError("Audio URL redirect is invalid")
                current_url = urljoin(current_url, location)
                continue
            if response.status != 200:
                raise RequestValidationError(
                    "Audio download returned a non-success status"
                )

            content_length = response.getheader("Content-Length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise RequestValidationError(
                        "Audio download returned an invalid size"
                    ) from exc
                if declared_size < 0 or declared_size > MAX_AUDIO_BYTES:
                    raise RequestValidationError("Audio file exceeds the size limit")

            total = 0
            with destination.open("wb") as output:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RequestValidationError("Audio download timed out")
                    if connection.sock is not None:
                        connection.sock.settimeout(remaining)
                    chunk = response.read1(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise RequestValidationError("Audio file exceeds the size limit")
                    output.write(chunk)
            if total == 0:
                raise RequestValidationError("Audio file is empty")
            return
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            raise RequestValidationError("Audio download failed") from exc
        finally:
            connection.close()


def _decode_audio(value: Any, destination: Path) -> None:
    if not isinstance(value, str) or not value:
        raise RequestValidationError("Audio base64 must be a non-empty string")
    maximum_encoded_length = 4 * ((MAX_AUDIO_BYTES + 2) // 3)
    if len(value) > maximum_encoded_length:
        raise RequestValidationError("Audio file exceeds the size limit")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestValidationError("Audio base64 is invalid") from exc
    if not raw:
        raise RequestValidationError("Audio file is empty")
    if len(raw) > MAX_AUDIO_BYTES:
        raise RequestValidationError("Audio file exceeds the size limit")
    destination.write_bytes(raw)


def _materialize_audio(
    data: dict[str, Any], prefix: str, directory: Path
) -> Path | None:
    url_field = f"{prefix}_audio_url"
    base64_field = f"{prefix}_audio_base64"
    extension_field = f"{prefix}_audio_extension"
    has_url = data.get(url_field) is not None
    has_base64 = data.get(base64_field) is not None
    if has_url == has_base64:
        if prefix == "speaker":
            raise RequestValidationError(
                "Provide exactly one of speaker_audio_url or speaker_audio_base64"
            )
        if not has_url:
            if data.get(extension_field) is not None:
                raise RequestValidationError(
                    "emotion_audio_extension requires an emotion audio source"
                )
            return None
        raise RequestValidationError(
            "Provide at most one of emotion_audio_url or emotion_audio_base64"
        )

    extension = _validate_extension(data.get(extension_field), extension_field)
    source_path = directory / f"{prefix}_source.{extension}"
    if has_url:
        _download_audio(data[url_field], source_path)
    else:
        _decode_audio(data[base64_field], source_path)

    normalized_path = directory / f"{prefix}_normalized.wav"
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-t",
        "15",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(NORMALIZED_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(normalized_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=max(DOWNLOAD_TIMEOUT_SECONDS, 30),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RequestValidationError("Audio could not be decoded") from exc
    if not normalized_path.is_file() or normalized_path.stat().st_size <= 44:
        raise RequestValidationError("Audio contains no decodable samples")
    return normalized_path


def _validate_emotion_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != EMOTION_VECTOR_SIZE:
        raise RequestValidationError("emotion_vector must contain exactly 8 numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RequestValidationError("emotion_vector must contain exactly 8 numbers")
        number = float(item)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise RequestValidationError(
                "emotion_vector values must be finite numbers from 0 to 1"
            )
        result.append(number)
    return result


def _validate_request(data: dict[str, Any]) -> dict[str, Any]:
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RequestValidationError("text must be a non-empty string")
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        raise RequestValidationError(
            f"text exceeds the maximum length of {MAX_TEXT_LENGTH} characters"
        )

    alpha = data.get("emotion_alpha", 1.0)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise RequestValidationError("emotion_alpha must be a number from 0 to 1")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise RequestValidationError("emotion_alpha must be a number from 0 to 1")

    seed = data.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RequestValidationError("seed must be an integer")
        if not 0 <= seed <= 2**32 - 1:
            raise RequestValidationError("seed must be between 0 and 4294967295")

    emotion_text = _optional_text(data, "emotion_text")
    if emotion_text is not None and len(emotion_text) > MAX_TEXT_LENGTH:
        raise RequestValidationError(
            f"emotion_text exceeds the maximum length of {MAX_TEXT_LENGTH} characters"
        )
    emotion_vector = _validate_emotion_vector(data.get("emotion_vector"))
    use_text_emotion = _optional_bool(data, "use_text_emotion", False)
    if use_text_emotion and emotion_text is None:
        emotion_text = text
    if emotion_text is not None:
        use_text_emotion = True
    if emotion_vector is not None and use_text_emotion:
        raise RequestValidationError(
            "emotion_vector and text emotion control cannot be combined"
        )

    return {
        "text": text,
        "emotion_text": emotion_text,
        "emotion_vector": emotion_vector,
        "emotion_alpha": alpha,
        "use_text_emotion": use_text_emotion,
        "seed": seed,
        "use_random": _optional_bool(data, "use_random", False),
        "verbose": _optional_bool(data, "verbose", False),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_id = str(job.get("id", "unknown"))
    try:
        request = _require_object(job.get("input"), "input")
        options = _validate_request(request)
        effective_seed = options["seed"]
        if effective_seed is None:
            effective_seed = secrets.randbits(32) if options["use_random"] else 0

        LOGGER.info(
            "Job %s accepted text_chars=%d text_emotion=%s vector_emotion=%s "
            "random=%s seed=%d",
            job_id,
            len(options["text"]),
            options["use_text_emotion"],
            options["emotion_vector"] is not None,
            options["use_random"],
            effective_seed,
        )

        with tempfile.TemporaryDirectory(prefix="indextts2-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            speaker_path = _materialize_audio(request, "speaker", temp_dir)
            emotion_path = _materialize_audio(request, "emotion", temp_dir)
            if emotion_path is not None and (
                options["emotion_vector"] is not None
                or options["use_text_emotion"]
            ):
                raise RequestValidationError(
                    "Emotion audio cannot be combined with vector or text emotion control"
                )
            output_path = temp_dir / "output.wav"

            with INFERENCE_LOCK, torch.inference_mode():
                _seed_everything(effective_seed)
                # spk_audio_prompt is IndexTTS2's zero-shot speaker/voice reference.
                # emo_audio_prompt, emo_vector, use_emo_text/emo_text, and emo_alpha
                # are the emotion controls in the official IndexTTS2 API.
                result = MODEL.infer(
                    spk_audio_prompt=str(speaker_path),
                    text=options["text"],
                    output_path=str(output_path),
                    emo_audio_prompt=str(emotion_path) if emotion_path else None,
                    emo_alpha=options["emotion_alpha"],
                    emo_vector=options["emotion_vector"],
                    use_emo_text=options["use_text_emotion"],
                    emo_text=options["emotion_text"],
                    use_random=options["use_random"],
                    verbose=options["verbose"],
                )
            if result is None or not output_path.is_file():
                raise RuntimeError("Inference did not produce audio")
            audio = output_path.read_bytes()
            if not audio:
                raise RuntimeError("Inference produced an empty audio file")

        LOGGER.info("Job %s completed output_bytes=%d", job_id, len(audio))
        return {
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "content_type": "audio/wav",
            "size_bytes": len(audio),
            "seed": effective_seed,
        }
    except RequestValidationError as exc:
        LOGGER.exception("Job %s rejected", job_id)
        raise RuntimeError(str(exc)) from None
    except Exception:
        LOGGER.exception("Job %s failed", job_id)
        raise RuntimeError("Audio generation failed; check worker logs") from None


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
