from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


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


def download_models(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    token = _token()
    print(f"Downloading {MODEL_REPO}@{MODEL_REVISION} to {model_dir}")
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=model_dir,
        token=token,
    )

    cache_dir = model_dir / "hf_cache"
    w2v_dir = cache_dir / "w2v-bert-2.0"
    snapshot_download(
        repo_id=W2V_REPO,
        revision=W2V_REVISION,
        local_dir=w2v_dir,
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
        token=token,
    )

    nested_semantic_codec = Path(
        hf_hub_download(
            repo_id=MASKGCT_REPO,
            revision=MASKGCT_REVISION,
            filename="semantic_codec/model.safetensors",
            local_dir=cache_dir,
            token=token,
        )
    )
    shutil.copyfile(
        nested_semantic_codec, cache_dir / "semantic_codec_model.safetensors"
    )

    hf_hub_download(
        repo_id=CAMPPLUS_REPO,
        revision=CAMPPLUS_REVISION,
        filename="campplus_cn_common.bin",
        local_dir=cache_dir,
        token=token,
    )

    bigvgan_dir = cache_dir / "bigvgan"
    snapshot_download(
        repo_id=BIGVGAN_REPO,
        revision=BIGVGAN_REVISION,
        local_dir=bigvgan_dir,
        allow_patterns=["config.json", "bigvgan_generator.pt"],
        token=token,
    )
    print("All pinned IndexTTS2 model assets downloaded")


if __name__ == "__main__":
    target = Path(
        os.getenv("MODEL_DIR", "/runpod-volume/indextts2/checkpoints")
    ).resolve()
    download_models(target)
