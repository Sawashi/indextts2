from __future__ import annotations

import base64
import binascii
import errno
import fcntl
import http.client
import ipaddress
import logging
import math
import os
import random
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
LOGGER = logging.getLogger("indextts2-worker")


def _startup_banner() -> None:
    print("=== IndexTTS2 Runpod worker startup ===", flush=True)
    get_uid = getattr(os, "getuid", lambda: "unavailable")
    get_gid = getattr(os, "getgid", lambda: "unavailable")
    model_dir = os.getenv("MODEL_DIR", "/runpod-volume/indextts2/checkpoints")
    config_path = os.getenv("CONFIG_PATH", f"{model_dir}/config.yaml")

    print(
        f"uid={get_uid()} gid={get_gid()} hostname={socket.gethostname()} "
        f"python={sys.version.replace(os.linesep, ' ')}",
        flush=True,
    )
    print(f"MODEL_DIR={model_dir}", flush=True)
    print(f"CONFIG_PATH={config_path}", flush=True)
    print(
        f"MODEL_DOWNLOAD_ON_START={os.getenv('MODEL_DOWNLOAD_ON_START', 'true')}",
        flush=True,
    )
    print(
        f"MODEL_LOCK_TIMEOUT_SECONDS={os.getenv('MODEL_LOCK_TIMEOUT_SECONDS', '1800')}",
        flush=True,
    )
    print(
        "HF_DOWNLOAD_TIMEOUT_SECONDS="
        f"{os.getenv('HF_DOWNLOAD_TIMEOUT_SECONDS', '600')}",
        flush=True,
    )
    print(
        f"HF_DOWNLOAD_RETRIES={os.getenv('HF_DOWNLOAD_RETRIES', '5')}",
        flush=True,
    )
    print(
        f"HF_DOWNLOAD_BACKOFF_SECONDS={os.getenv('HF_DOWNLOAD_BACKOFF_SECONDS', '5')}",
        flush=True,
    )
    print(f"HF_TOKEN present={bool(os.getenv('HF_TOKEN'))}", flush=True)

    volume_path = "/runpod-volume"
    print("Checking /runpod-volume mount and free space...", flush=True)
    volume_exists = os.path.exists(volume_path)
    volume_mounted = os.path.ismount(volume_path)
    try:
        free_bytes = shutil.disk_usage(volume_path).free if volume_exists else None
        free_disk = (
            f"{free_bytes} bytes ({free_bytes / (1024**3):.2f} GiB)"
            if free_bytes is not None
            else "unavailable"
        )
    except OSError as exc:
        free_disk = f"unavailable ({type(exc).__name__})"
    print(
        f"/runpod-volume exists={volume_exists} ismount={volume_mounted} "
        f"free_disk={free_disk}",
        flush=True,
    )

    print("Loading PyTorch for CUDA diagnostics...", flush=True)
    try:
        import torch as torch_for_banner

        cuda_available = torch_for_banner.cuda.is_available()
        gpu_name = (
            torch_for_banner.cuda.get_device_name(0)
            if cuda_available
            else "unavailable"
        )
        print(
            f"torch.cuda.is_available()={cuda_available} gpu_name={gpu_name}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"PyTorch CUDA diagnostics failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    print("=== Startup diagnostics complete; model loading is deferred ===", flush=True)


_startup_banner()


import numpy as np  # noqa: E402
import runpod  # noqa: E402
import torch  # noqa: E402
from download_models import MODEL_REVISION_MARKER, download_models  # noqa: E402


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


VOLUME_ROOT = Path("/runpod-volume")
MODEL_ROOT = VOLUME_ROOT / "indextts2"
DOWNLOAD_LOCK_PATH = MODEL_ROOT / ".download.lock"
DOWNLOAD_MARKER_NAME = ".download-complete"
MODEL_DIR = Path(
    os.getenv("MODEL_DIR", "/runpod-volume/indextts2/checkpoints")
).resolve()
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", str(MODEL_DIR / "config.yaml"))).resolve()
MODEL_DOWNLOAD_ON_START = _env_bool("MODEL_DOWNLOAD_ON_START", True)
USE_FP16 = _env_bool("USE_FP16", True)
USE_DEEPSPEED = _env_bool("USE_DEEPSPEED", False)
USE_CUDA_KERNEL = _env_bool("USE_CUDA_KERNEL", False)
MAX_TEXT_LENGTH = _env_positive_int("MAX_TEXT_LENGTH", 5_000)
MAX_AUDIO_BYTES = _env_positive_int("MAX_AUDIO_BYTES", 25 * 1024 * 1024)
DOWNLOAD_TIMEOUT_SECONDS = _env_positive_int("DOWNLOAD_TIMEOUT_SECONDS", 30)
MODEL_LOCK_TIMEOUT_SECONDS = _env_positive_int("MODEL_LOCK_TIMEOUT_SECONDS", 1_800)


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
        "W2V-BERT preprocessor": cache / "w2v-bert-2.0" / "preprocessor_config.json",
        "semantic codec": cache / "semantic_codec_model.safetensors",
        "CAMPPlus": cache / "campplus_cn_common.bin",
        "BigVGAN config": cache / "bigvgan" / "config.json",
        "BigVGAN model": cache / "bigvgan" / "bigvgan_generator.pt",
    }


def _model_files_ready(model_dir: Path, config_path: Path) -> bool:
    marker = model_dir / DOWNLOAD_MARKER_NAME
    try:
        marker_matches = marker.read_text(encoding="ascii") == MODEL_REVISION_MARKER
    except (OSError, UnicodeError):
        return False
    return marker_matches and all(
        path.is_file()
        for path in _required_model_paths(model_dir, config_path).values()
    )


def _assert_network_volume() -> None:
    error_message = (
        "Runpod network volume at /runpod-volume is unavailable: volume not "
        "mounted or permissions wrong; attach volume in endpoint settings"
    )
    if not VOLUME_ROOT.is_dir() or not os.path.ismount(VOLUME_ROOT):
        raise RuntimeError(error_message)
    try:
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".write-test-", dir=MODEL_ROOT
        ) as write_test:
            write_test.write(b"runpod-volume-write-test\n")
            write_test.flush()
    except OSError as exc:
        raise RuntimeError(error_message) from exc


def _prepare_runtime_directories() -> None:
    tagger_cache = (
        Path(__file__).resolve().parent / "indextts" / "utils" / "tagger_cache"
    )
    try:
        tagger_cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Could not create runtime cache directory {tagger_cache}"
        ) from exc


def _write_download_marker(download_dir: Path) -> None:
    marker = download_dir / DOWNLOAD_MARKER_NAME
    temporary_marker = download_dir / (
        f".{DOWNLOAD_MARKER_NAME}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary_marker.open("w", encoding="ascii", newline="") as output:
            output.write(MODEL_REVISION_MARKER)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_marker, marker)
    finally:
        try:
            temporary_marker.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("Could not remove temporary marker %s", temporary_marker)


def _acquire_download_lock(lock_file: Any) -> None:
    started = time.monotonic()
    deadline = started + MODEL_LOCK_TIMEOUT_SECONDS
    next_progress_log = started + 10
    LOGGER.info(
        "Waiting up to %d seconds for model download lock %s",
        MODEL_LOCK_TIMEOUT_SECONDS,
        DOWNLOAD_LOCK_PATH,
    )
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            LOGGER.info(
                "Acquired model download lock after %.1f seconds",
                time.monotonic() - started,
            )
            return
        except BlockingIOError:
            pass
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError("Could not acquire the model download lock") from exc

        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            raise RuntimeError(
                "Timed out after "
                f"{MODEL_LOCK_TIMEOUT_SECONDS} seconds waiting for model download "
                f"lock {DOWNLOAD_LOCK_PATH}; another worker may be stuck"
            )
        if now >= next_progress_log:
            LOGGER.info(
                "Still waiting for lock... path=%s elapsed_seconds=%.0f "
                "remaining_seconds=%.0f",
                DOWNLOAD_LOCK_PATH,
                now - started,
                remaining,
            )
            next_progress_log = now + 10
        time.sleep(min(1.0, remaining))


def _publish_download(download_dir: Path) -> None:
    stale_dir: Path | None = None
    if MODEL_DIR.exists():
        stale_dir = MODEL_DIR.parent / (
            f".{MODEL_DIR.name}-incomplete-{os.getpid()}-{time.time_ns()}"
        )
        os.replace(MODEL_DIR, stale_dir)
    try:
        os.replace(download_dir, MODEL_DIR)
    except Exception:
        if stale_dir is not None and stale_dir.exists() and not MODEL_DIR.exists():
            os.replace(stale_dir, MODEL_DIR)
        raise
    if stale_dir is not None:
        try:
            shutil.rmtree(stale_dir)
        except OSError:
            LOGGER.warning("Could not remove old checkpoint directory %s", stale_dir)


def _recover_interrupted_downloads() -> None:
    stale_dirs = sorted(MODEL_DIR.parent.glob(f".{MODEL_DIR.name}-incomplete-*"))
    download_dirs = sorted(MODEL_DIR.parent.glob(f".{MODEL_DIR.name}-download-*"))

    if not MODEL_DIR.exists():
        for stale_dir in reversed(stale_dirs):
            if _model_files_ready(stale_dir, stale_dir / "config.yaml"):
                LOGGER.warning("Restoring interrupted checkpoint publication")
                os.replace(stale_dir, MODEL_DIR)
                stale_dirs.remove(stale_dir)
                break

    for orphan in [*stale_dirs, *download_dirs]:
        try:
            shutil.rmtree(orphan)
            LOGGER.info("Removed interrupted model download directory %s", orphan)
        except OSError:
            LOGGER.warning("Could not remove orphaned model directory %s", orphan)


def _prepare_model_files() -> None:
    _assert_network_volume()
    if _model_files_ready(MODEL_DIR, CONFIG_PATH):
        LOGGER.info("Pinned model checkpoints are ready at %s", MODEL_DIR)
        return

    try:
        lock_file = DOWNLOAD_LOCK_PATH.open("a+")
    except OSError as exc:
        raise RuntimeError("Could not open the model download lock") from exc

    with lock_file:
        _acquire_download_lock(lock_file)
        try:
            _recover_interrupted_downloads()
            if _model_files_ready(MODEL_DIR, CONFIG_PATH):
                LOGGER.info("Another worker completed the model download")
                return
            if not MODEL_DOWNLOAD_ON_START:
                missing = [
                    label
                    for label, path in _required_model_paths(
                        MODEL_DIR, CONFIG_PATH
                    ).items()
                    if not path.is_file()
                ]
                marker = MODEL_DIR / DOWNLOAD_MARKER_NAME
                try:
                    marker_matches = (
                        marker.read_text(encoding="ascii") == MODEL_REVISION_MARKER
                    )
                except (OSError, UnicodeError):
                    marker_matches = False
                if not marker_matches:
                    missing.append(f"{DOWNLOAD_MARKER_NAME} revision marker")
                raise RuntimeError(
                    "Model checkpoints are incomplete and "
                    "MODEL_DOWNLOAD_ON_START is false; missing: " + ", ".join(missing)
                )

            expected_config = (MODEL_DIR / "config.yaml").resolve()
            if CONFIG_PATH != expected_config:
                raise RuntimeError(
                    "Automatic model download requires CONFIG_PATH to be "
                    f"{expected_config}"
                )

            MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            download_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{MODEL_DIR.name}-download-", dir=MODEL_DIR.parent
                )
            )
            LOGGER.info("Downloading pinned model checkpoints to %s", download_dir)
            try:
                download_models(download_dir)
                downloaded_required = _required_model_paths(
                    download_dir, download_dir / "config.yaml"
                )
                missing = [
                    label
                    for label, path in downloaded_required.items()
                    if not path.is_file()
                ]
                if missing:
                    raise RuntimeError(
                        "Model download completed with missing files: "
                        + ", ".join(missing)
                    )
                _write_download_marker(download_dir)
                _publish_download(download_dir)
            finally:
                if download_dir.exists():
                    try:
                        shutil.rmtree(download_dir)
                    except OSError:
                        LOGGER.exception(
                            "Could not remove failed model download directory %s",
                            download_dir,
                        )

            if not _model_files_ready(MODEL_DIR, CONFIG_PATH):
                raise RuntimeError(
                    "Published model checkpoints failed the completeness check"
                )
            LOGGER.info("Model checkpoints published at %s", MODEL_DIR)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_model() -> Any:
    _prepare_runtime_directories()
    _prepare_model_files()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but no NVIDIA GPU is available")
    if USE_DEEPSPEED:
        try:
            import deepspeed  # noqa: F401
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "USE_DEEPSPEED is enabled but DeepSpeed is not installed correctly"
            ) from exc

    LOGGER.info("Importing the IndexTTS2 inference runtime")
    try:
        from indextts.infer_v2 import IndexTTS2
    except Exception:
        LOGGER.exception("IndexTTS2 runtime import failed")
        raise RuntimeError(
            "IndexTTS2 runtime import failed; check worker logs"
        ) from None

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
        raise RuntimeError(
            "IndexTTS2 initialization failed; check worker logs"
        ) from None
    if USE_CUDA_KERNEL and not model.use_cuda_kernel:
        raise RuntimeError(
            "USE_CUDA_KERNEL is enabled but the custom CUDA kernel did not load"
        )
    LOGGER.info("IndexTTS2 initialized")
    return model


MODEL = None
MODEL_INIT_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()


def get_model() -> Any:
    global MODEL

    if MODEL is not None:
        return MODEL
    with MODEL_INIT_LOCK:
        if MODEL is None:
            started = time.monotonic()
            LOGGER.info("Starting lazy IndexTTS2 model initialization")
            MODEL = _load_model()
            LOGGER.info(
                "Lazy IndexTTS2 model initialization completed in %.1f seconds",
                time.monotonic() - started,
            )
        return MODEL


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
        raise RequestValidationError(
            "Audio URL resolved to an invalid address"
        ) from exc
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
                        raise RequestValidationError(
                            "Audio file exceeds the size limit"
                        )
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
            raise RequestValidationError(
                "emotion_vector must contain exactly 8 numbers"
            )
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
                options["emotion_vector"] is not None or options["use_text_emotion"]
            ):
                raise RequestValidationError(
                    "Emotion audio cannot be combined with vector or text emotion control"
                )
            output_path = temp_dir / "output.wav"
            model = get_model()

            with INFERENCE_LOCK, torch.inference_mode():
                _seed_everything(effective_seed)
                # spk_audio_prompt is IndexTTS2's zero-shot speaker/voice reference.
                # emo_audio_prompt, emo_vector, use_emo_text/emo_text, and emo_alpha
                # are the emotion controls in the official IndexTTS2 API.
                result = model.infer(
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
    LOGGER.info("Validating the Runpod network volume before accepting jobs")
    _assert_network_volume()
    LOGGER.info("Runpod network volume mount and write test passed")
    LOGGER.info(
        "Starting Runpod serverless handler; model initialization is deferred "
        "until the first job"
    )
    runpod.serverless.start({"handler": handler})
