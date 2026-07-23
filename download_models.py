from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        requirement = "greater than zero" if minimum == 1 else "zero or greater"
        raise RuntimeError(f"{name} must be {requirement}")
    return value


HF_DOWNLOAD_TIMEOUT_SECONDS = _env_int("HF_DOWNLOAD_TIMEOUT_SECONDS", 600, minimum=1)
HF_DOWNLOAD_RETRIES = _env_int("HF_DOWNLOAD_RETRIES", 5, minimum=0)
HF_DOWNLOAD_BACKOFF_SECONDS = _env_int("HF_DOWNLOAD_BACKOFF_SECONDS", 5, minimum=0)

# huggingface_hub reads these request timeouts while it is imported.
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(HF_DOWNLOAD_TIMEOUT_SECONDS)
os.environ["HF_HUB_ETAG_TIMEOUT"] = str(HF_DOWNLOAD_TIMEOUT_SECONDS)

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
LOGGER = logging.getLogger("indextts2-download")
DOWNLOAD_PROGRESS_LOG_SECONDS = 30
T = TypeVar("T")


MODEL_REPO = "IndexTeam/IndexTTS-2"
MODEL_REVISION = "740dcaff396282ffb241903d150ac011cd4b1ede"
W2V_REPO = "facebook/w2v-bert-2.0"
W2V_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
MASKGCT_REPO = "amphion/MaskGCT"
MASKGCT_REVISION = "265c6cef07625665d0c28d2faafb1415562379dc"
CAMPPLUS_REPO = "funasr/campplus"
CAMPPLUS_REVISION = "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"
BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
BIGVGAN_REVISION = "633ff708ed5b74903e86ff1298cf4a98e921c513"
MODEL_REVISION_MARKER = "\n".join(
    [
        f"{MODEL_REPO}@{MODEL_REVISION}",
        f"{W2V_REPO}@{W2V_REVISION}",
        f"{MASKGCT_REPO}@{MASKGCT_REVISION}",
        f"{CAMPPLUS_REPO}@{CAMPPLUS_REVISION}",
        f"{BIGVGAN_REPO}@{BIGVGAN_REVISION}",
        "",
    ]
)


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _call_with_progress(repository: str, attempt: int, operation: Callable[[], T]) -> T:
    started = time.monotonic()
    finished = threading.Event()

    def log_progress() -> None:
        while not finished.wait(DOWNLOAD_PROGRESS_LOG_SECONDS):
            LOGGER.info(
                "Still downloading repository=%s attempt=%d elapsed_seconds=%.0f",
                repository,
                attempt,
                time.monotonic() - started,
            )

    progress_thread = threading.Thread(
        target=log_progress,
        name="hf-download-progress",
        daemon=True,
    )
    progress_thread.start()
    try:
        return operation()
    finally:
        finished.set()
        progress_thread.join(timeout=1)


def _wait_before_retry(repository: str, delay_seconds: int) -> None:
    if delay_seconds <= 0:
        return
    started = time.monotonic()
    deadline = started + delay_seconds
    LOGGER.info(
        "Waiting %d seconds before retrying repository=%s",
        delay_seconds,
        repository,
    )
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(10.0, remaining))
        remaining = deadline - time.monotonic()
        if remaining > 0:
            LOGGER.info(
                "Still waiting to retry repository=%s remaining_seconds=%.0f",
                repository,
                remaining,
            )


def _download_with_retries(
    repo_id: str, revision: str, operation: Callable[[], T]
) -> T:
    repository = f"{repo_id}@{revision}"
    total_started = time.monotonic()
    max_attempts = HF_DOWNLOAD_RETRIES + 1
    LOGGER.info(
        "Starting repository download repository=%s timeout_seconds=%d max_attempts=%d",
        repository,
        HF_DOWNLOAD_TIMEOUT_SECONDS,
        max_attempts,
    )
    for attempt in range(1, max_attempts + 1):
        attempt_started = time.monotonic()
        LOGGER.info(
            "Downloading repository=%s attempt=%d/%d",
            repository,
            attempt,
            max_attempts,
        )
        try:
            result = _call_with_progress(repository, attempt, operation)
        except Exception as exc:
            attempt_elapsed = time.monotonic() - attempt_started
            if attempt == max_attempts:
                LOGGER.error(
                    "Repository download failed repository=%s attempt=%d/%d "
                    "attempt_elapsed_seconds=%.1f total_elapsed_seconds=%.1f "
                    "error_type=%s error=%s",
                    repository,
                    attempt,
                    max_attempts,
                    attempt_elapsed,
                    time.monotonic() - total_started,
                    type(exc).__name__,
                    exc,
                )
                raise RuntimeError(
                    f"Hugging Face download failed for {repository} after "
                    f"{max_attempts} attempts"
                ) from exc

            delay_seconds = min(
                HF_DOWNLOAD_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                HF_DOWNLOAD_TIMEOUT_SECONDS,
            )
            LOGGER.warning(
                "Repository download attempt failed repository=%s "
                "attempt=%d/%d elapsed_seconds=%.1f error_type=%s error=%s "
                "retry_in_seconds=%d",
                repository,
                attempt,
                max_attempts,
                attempt_elapsed,
                type(exc).__name__,
                exc,
                delay_seconds,
            )
            _wait_before_retry(repository, delay_seconds)
            continue

        LOGGER.info(
            "Completed repository download repository=%s attempt=%d/%d "
            "attempt_elapsed_seconds=%.1f total_elapsed_seconds=%.1f",
            repository,
            attempt,
            max_attempts,
            time.monotonic() - attempt_started,
            time.monotonic() - total_started,
        )
        return result

    raise AssertionError("unreachable")


def download_models(model_dir: Path) -> None:
    started = time.monotonic()
    model_dir.mkdir(parents=True, exist_ok=True)
    token = _token()
    LOGGER.info(
        "Beginning pinned model downloads destination=%s token_present=%s",
        model_dir,
        bool(token),
    )
    _download_with_retries(
        MODEL_REPO,
        MODEL_REVISION,
        lambda: snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=model_dir,
            token=token,
            etag_timeout=HF_DOWNLOAD_TIMEOUT_SECONDS,
        ),
    )

    cache_dir = model_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    w2v_dir = cache_dir / "w2v-bert-2.0"
    _download_with_retries(
        W2V_REPO,
        W2V_REVISION,
        lambda: snapshot_download(
            repo_id=W2V_REPO,
            revision=W2V_REVISION,
            local_dir=w2v_dir,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
            ],
            token=token,
            etag_timeout=HF_DOWNLOAD_TIMEOUT_SECONDS,
        ),
    )

    nested_semantic_codec = Path(
        _download_with_retries(
            MASKGCT_REPO,
            MASKGCT_REVISION,
            lambda: hf_hub_download(
                repo_id=MASKGCT_REPO,
                revision=MASKGCT_REVISION,
                filename="semantic_codec/model.safetensors",
                local_dir=cache_dir,
                token=token,
                etag_timeout=HF_DOWNLOAD_TIMEOUT_SECONDS,
            ),
        )
    )
    shutil.copyfile(
        nested_semantic_codec, cache_dir / "semantic_codec_model.safetensors"
    )

    _download_with_retries(
        CAMPPLUS_REPO,
        CAMPPLUS_REVISION,
        lambda: hf_hub_download(
            repo_id=CAMPPLUS_REPO,
            revision=CAMPPLUS_REVISION,
            filename="campplus_cn_common.bin",
            local_dir=cache_dir,
            token=token,
            etag_timeout=HF_DOWNLOAD_TIMEOUT_SECONDS,
        ),
    )

    bigvgan_dir = cache_dir / "bigvgan"
    _download_with_retries(
        BIGVGAN_REPO,
        BIGVGAN_REVISION,
        lambda: snapshot_download(
            repo_id=BIGVGAN_REPO,
            revision=BIGVGAN_REVISION,
            local_dir=bigvgan_dir,
            allow_patterns=["config.json", "bigvgan_generator.pt"],
            token=token,
            etag_timeout=HF_DOWNLOAD_TIMEOUT_SECONDS,
        ),
    )
    LOGGER.info(
        "All pinned IndexTTS2 model assets downloaded elapsed_seconds=%.1f",
        time.monotonic() - started,
    )


if __name__ == "__main__":
    target = Path(
        os.getenv("MODEL_DIR", "/runpod-volume/indextts2/checkpoints")
    ).resolve()
    download_models(target)
