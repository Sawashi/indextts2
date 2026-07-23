# IndexTTS2 Runpod Serverless Worker

GPU queue worker for the official [IndexTTS2](https://github.com/index-tts/index-tts)
zero-shot text-to-speech model. The process registers its queue handler without
loading the model; the first request initializes it once, accepts a temporary
speaker reference recording, and returns a base64-encoded WAV file. There is no
web UI or HTTP framework; Runpod's Python SDK supplies the queue worker protocol.

> **Voice-cloning consent warning:** Use this worker only with a speaker's
> informed authorization or another valid legal basis. Do not impersonate,
> deceive, harass, or defraud. Keep consent records and comply with applicable
> biometric, privacy, publicity, copyright, and synthetic-media laws.

## Pinned Upstream Components

- IndexTTS2 source: `index-tts/index-tts` commit
  `13495845e3028f0bb6ca1462ad22aa0e76349e40`
- IndexTTS2 model: `IndexTeam/IndexTTS-2` revision
  `740dcaff396282ffb241903d150ac011cd4b1ede`
- Python: `3.11.13`; PyTorch `2.8.*` CUDA 12.8 from IndexTTS2's `uv.lock`
- Runpod SDK: `1.11.0`

`download_models.py` also pins the four auxiliary model repositories. Source and
Python dependencies are built into the image; model weights are stored on the
attached Runpod network volume. Updating IndexTTS2 requires reviewing its API,
lock file, checkpoint layout, licenses, and GPU behavior before changing
`INDEXTTS_REF`. Using mutable `main` in production is intentionally avoided.

## How Voice Cloning Works

IndexTTS2's official `IndexTTS2.infer` API receives the normalized reference
recording through `spk_audio_prompt`. This is zero-shot conditioning: no model is
trained or fine-tuned for the speaker. Input files are held in a
`TemporaryDirectory`, truncated to the same 15-second maximum used upstream,
normalized to mono 22.05 kHz PCM WAV, and deleted after the request.

Emotion controls map directly to the official API:

- `emotion_audio_*` becomes `emo_audio_prompt`.
- `emotion_vector` becomes `emo_vector` and must contain eight values ordered
  `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.
- `emotion_text` or `use_text_emotion: true` enables `use_emo_text`/`emo_text`.
- `emotion_alpha` becomes `emo_alpha` and must be from 0 to 1.

Audio, vector, and text emotion modes are mutually exclusive to avoid ambiguous
inputs. With no explicit emotion control, upstream uses the speaker recording as
its emotion reference. `use_random` defaults to `false`, preserving the
speaker-similar emotion prototype. The effective default seed is `0`; when
`use_random` is explicitly true and no seed is supplied, the worker creates and
returns a random 32-bit seed.

## Prerequisites

- A GitHub account and repository
- A Runpod account with Serverless access and a Runpod API key
- A public HTTPS speaker-audio URL, or a base64 recording
- Docker with NVIDIA Container Toolkit for local GPU testing
- A writable Runpod Serverless network volume with enough capacity for all
  primary and auxiliary model files

The native Runpod GitHub builder currently has a 30-minute Docker build-step
limit, 160-minute total build window, and 80 GB image limit. Checkpoints are not
downloaded during the image build, keeping the worker image smaller and model
storage persistent across image deployments.

## GitHub Setup

Create an empty GitHub repository, then push this project:

```bash
git init
git add .
git commit -m "Add Runpod IndexTTS2 serverless worker"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Do not commit Hugging Face or Runpod tokens. The default model assets are public,
so the Runpod build needs no token.

## Deploy From GitHub

1. In Runpod, open **Settings > Connections**, connect GitHub, and grant access
   to this repository.
2. Select **New Endpoint**, then **Import Git Repository** (the deploy-from-GitHub
   workflow).
3. Choose this repository and the `main` branch.
4. Set the Dockerfile path to `Dockerfile` and the build context to the repository
   root. No build arguments are required for the pinned default.
5. Wait for the clone, build, upload, test, and deployment stages in **Builds**.
6. Create a GitHub release when you want Runpod to trigger a new build; ordinary
   pushes do not automatically redeploy an existing endpoint.

Build locally with:

```bash
docker buildx build --platform linux/amd64 -t indextts2-runpod:13495845 .
```

Do not use a token as a Docker `ARG` or commit it to the repository. The pinned
assets are public. If a token is required in the future, set `HF_TOKEN` as a
Runpod endpoint environment variable so it is available only at runtime.

## Attach A Network Volume

1. In the Runpod console, create a network volume under **Storage** in the data
   center where this Serverless endpoint will run. Allocate enough space for the
   pinned IndexTTS2 model and all auxiliary models, plus room for a temporary
   second copy during future replacement downloads.
2. In the Serverless endpoint configuration, select that network volume. Runpod
   mounts an attached Serverless network volume inside each worker at
   `/runpod-volume`.
3. Keep the default paths for automatic download:
   `/runpod-volume/indextts2/checkpoints` and
   `/runpod-volume/indextts2/checkpoints/config.yaml`.
4. The container runs as root and performs a startup write test. Startup fails
   immediately with a clear error if `/runpod-volume` is absent or not writable.

Network volumes are tied to a specific Runpod data center. The endpoint can use
only GPU types with availability in that same data center, so check GPU capacity
before creating or attaching the volume. Moving to another data center requires
a different volume and another first-start download.

Persistent network-volume storage is billed independently of active workers and
continues to incur storage cost when active workers are set to zero. Review
Runpod's current storage pricing and delete volumes that are no longer needed.

### First Worker Start

At process startup, the worker immediately logs its UID/GID, hostname, Python,
volume status and free space, CUDA status, GPU name, and non-secret configuration.
It write-tests the network volume and then starts the Runpod queue handler without
loading IndexTTS2.

On the first request, the worker checks the completion marker and every required
checkpoint and verifies that the marker contains the currently pinned model
revisions. If anything is absent or outdated and `MODEL_DOWNLOAD_ON_START=true`,
one worker acquires `/runpod-volume/indextts2/.download.lock`, downloads all
pinned assets into a temporary sibling directory, validates them, writes
`.download-complete`, and atomically publishes the directory as `checkpoints`.
Other workers wait up to `MODEL_LOCK_TIMEOUT_SECONDS` on the filesystem lock and
then reuse the completed files. Under the same lock, interrupted temporary
downloads are removed and an interrupted publication is restored when possible.

The first request on a new worker is therefore substantially longer and depends
on Hugging Face and network-volume throughput. Downloads use request timeouts,
exponential-backoff retries, and periodic progress logs. Later cold starts skip
downloading when the marker and required files exist, but still load the model
from the volume into GPU memory. The initialized model is reused by later jobs.

## Recommended Endpoint Settings

- Endpoint type: **Queue**
- Jobs per worker / per-worker concurrency: **1**
- Active workers: **0** for testing, or **1** for lower latency
- Max workers: **1** initially
- GPUs per worker: **1**
- GPU: start with **24 GB VRAM or higher**
- Execution timeout: **3600 seconds** for the first download and model load
- FP16: enabled (`USE_FP16=true`)
- DeepSpeed: disabled (`USE_DEEPSPEED=false`)
- Custom CUDA kernels: disabled (`USE_CUDA_KERNEL=false`)

The process-level lock also prevents overlapping inference if platform settings
are changed accidentally. Increase max workers, not jobs per worker, to add
parallel capacity.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_DIR` | `/runpod-volume/indextts2/checkpoints` | Persistent checkpoint and auxiliary-model directory |
| `CONFIG_PATH` | `/runpod-volume/indextts2/checkpoints/config.yaml` | Official config; automatic download requires this default layout |
| `MODEL_DOWNLOAD_ON_START` | `true` | Download pinned checkpoints during first model initialization when incomplete |
| `MODEL_LOCK_TIMEOUT_SECONDS` | `1800` | Maximum wait for another worker's checkpoint-download lock |
| `HF_DOWNLOAD_TIMEOUT_SECONDS` | `600` | Hugging Face metadata and file request timeout |
| `HF_DOWNLOAD_RETRIES` | `5` | Retries after a failed Hugging Face repository download attempt |
| `HF_DOWNLOAD_BACKOFF_SECONDS` | `5` | Initial exponential retry delay |
| `USE_FP16` | `true` | Enable FP16 where supported |
| `USE_DEEPSPEED` | `false` | Opt in to DeepSpeed; not installed by default |
| `USE_CUDA_KERNEL` | `false` | Opt in to BigVGAN custom CUDA kernel; not built by default |
| `MAX_TEXT_LENGTH` | `5000` | Maximum input characters |
| `MAX_AUDIO_BYTES` | `26214400` | Maximum encoded/downloaded bytes per audio input |
| `DOWNLOAD_TIMEOUT_SECONDS` | `30` | Separate connect/read timeout and ffmpeg minimum timeout |
| `LOG_LEVEL` | `INFO` | Python log level |

The default image intentionally omits DeepSpeed and custom-kernel extras. Setting
either opt-in variable to `true` also requires building an image with the matching
upstream optional dependency.

## Request Schema

Required:

- `text`: non-empty string, at most `MAX_TEXT_LENGTH` characters
- Exactly one of `speaker_audio_url` or `speaker_audio_base64`

Optional:

- `speaker_audio_extension`: `wav`, `mp3`, `flac`, `m4a`, or `ogg`; default `wav`
- One of `emotion_audio_url` / `emotion_audio_base64`, plus optional extension
- `emotion_vector`: exactly eight finite values from 0 to 1
- `emotion_text` (also limited by `MAX_TEXT_LENGTH`), `use_text_emotion`,
  `emotion_alpha`
- `seed`: integer from 0 through `4294967295`
- `use_random`, `verbose`: booleans

URLs must use HTTPS. Every DNS result and the connected peer are checked for a
public address, and the TLS connection is pinned to a validated address to
prevent DNS rebinding. Redirects are followed manually and revalidated. Localhost,
private, loopback, link-local, reserved, and metadata-service addresses are
rejected. Downloads use size limits and connect/read timeouts. Do not weaken
these controls for endpoints receiving untrusted input.

Runpod currently limits `/run` request payloads to 10 MB and `/runsync` payloads
to 20 MB. Base64 adds about 33 percent, so use HTTPS input for recordings near
the worker's 25 MB safety limit; large base64 inputs can be rejected by Runpod
before reaching the handler.

## Call `/run`

```bash
curl -X POST \
  "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "This is generated using a cloned reference voice.",
      "speaker_audio_url": "https://example.com/reference.wav",
      "emotion_text": "Friendly and conversational",
      "emotion_alpha": 0.6,
      "use_random": false,
      "seed": 42
    }
  }'
```

The asynchronous response contains a job `id`, normally with `IN_QUEUE` status.

## Poll `/status`

```bash
curl \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  "https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID"
```

Poll until `COMPLETED`, `FAILED`, `CANCELLED`, or `TIMED_OUT`. Clients should
tolerate both `IN_PROGRESS` and `RUNNING` as active states.

## Call `/runsync`

```bash
curl -X POST \
  "https://api.runpod.ai/v2/ENDPOINT_ID/runsync?wait=300000" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @test_input.json
```

The HTTP wait is not the endpoint execution timeout. If synthesis outlasts the
sync wait, use the returned job ID and poll `/status`.

Successful output is nested under Runpod's `output` field:

```json
{
  "output": {
    "audio_base64": "UklGR...",
    "content_type": "audio/wav",
    "size_bytes": 123456,
    "seed": 42
  },
  "status": "COMPLETED"
}
```

## Python Submit, Poll, And Decode

```python
import base64
import os
import time

import requests

endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
api_key = os.environ["RUNPOD_API_KEY"]
base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
headers = {"Authorization": f"Bearer {api_key}"}
payload = {
    "input": {
        "text": "This is generated using a cloned reference voice.",
        "speaker_audio_url": "https://example.com/reference.wav",
        "emotion_text": "Friendly and conversational",
        "emotion_alpha": 0.6,
        "use_random": False,
        "seed": 42,
    }
}

submitted = requests.post(
    f"{base_url}/run", headers=headers, json=payload, timeout=30
)
submitted.raise_for_status()
job_id = submitted.json()["id"]

deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    response = requests.get(
        f"{base_url}/status/{job_id}", headers=headers, timeout=30
    )
    response.raise_for_status()
    job = response.json()
    status = job["status"]
    if status == "COMPLETED":
        audio = base64.b64decode(job["output"]["audio_base64"], validate=True)
        with open("output.wav", "wb") as output_file:
            output_file.write(audio)
        break
    if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
        raise RuntimeError(f"Runpod job ended with status {status}")
    time.sleep(2)
else:
    raise TimeoutError("Timed out waiting for the Runpod job")
```

The API key is read from the environment and is never logged.

## Local Docker Testing

Build for Runpod's architecture:

```bash
docker build --platform linux/amd64 -t indextts2-runpod:13495845 .
```

Create a writable directory to simulate the mounted volume:

```bash
mkdir -p .local-runpod-volume
```

Run the SDK's local test mode with an NVIDIA GPU and that directory mounted at
the same path used by Runpod:

```bash
docker run --rm --gpus all \
  -e RUNPOD_DEBUG_LEVEL=INFO \
  -v "$(pwd)/test_input.json:/app/index-tts/test_input.json:ro" \
  -v "$(pwd)/.local-runpod-volume:/runpod-volume" \
  indextts2-runpod:13495845
```

Replace the placeholder URL in `test_input.json` with a reachable HTTPS audio
file before testing. Do not use a mutable `latest` tag for production builds.

## Troubleshooting

### CUDA Errors

- Confirm the worker has an NVIDIA GPU and the container runtime exposes it.
- Check startup logs for the GPU name and `torch.cuda.is_available()` failure.
- Keep the CUDA 12.8 runtime aligned with the pinned PyTorch 2.8 CUDA 12.8 wheels.
- Leave custom CUDA kernels disabled unless you intentionally add and test the
  upstream acceleration extra on the selected GPU architecture.

### Missing Checkpoints

- Confirm a network volume is attached at `/runpod-volume` and is writable.
- Confirm `MODEL_DIR` and `CONFIG_PATH` match the persistent volume paths.
- Check startup logs for Hugging Face download or volume-capacity failures.
- Ensure the volume has room for a temporary download before atomic publication.
- If automatic download is disabled, provision every pinned primary and
  auxiliary file and `.download-complete` before starting the worker.

### Out Of Memory

- Start with a 24 GB or larger GPU and retain FP16.
- Ensure jobs per worker is 1. The lock serializes inference but cannot reduce
  the model's baseline memory.
- Shorten input text or split long synthesis into multiple jobs.
- Move to a larger GPU before enabling optional acceleration components.

### Cold Starts

Initialization loads several neural networks from the attached volume. The first
worker on a new volume must also download and atomically publish all pinned model
assets, making that first cold start much longer. Active workers set to zero
minimize compute cost but incur model-load latency after scale-up; volume storage
cost continues while workers are stopped. Keep one active worker for
latency-sensitive production traffic, and monitor startup time before raising
scale limits.

## Production Security And Delivery

Do not expose this endpoint or its Runpod API key directly to untrusted browser
clients. Put a controlled backend in front of it and:

- Keep the Runpod API key server-side and rotate it regularly.
- Authenticate and authorize end users.
- Apply per-user rate limits, quotas, and request-size limits.
- Log requests without storing raw voice recordings or secrets.
- Keep verifiable speaker-consent records.
- Add abuse detection, moderation, and impersonation safeguards.
- Watermark generated audio or disclose synthetic origin where appropriate.
- Use short-lived signed URLs for input and output objects.

Base64 increases payload size by roughly one third. It is suitable for the
initial API, but production systems should upload generated WAV files to private
S3-compatible storage and return a short-lived signed URL instead. Runpod's
request/result size and retention limits also make object storage preferable for
long audio.

## Licensing

Read [LICENSE-NOTICE.md](LICENSE-NOTICE.md) before use. Source code, primary
weights, auxiliary weights, CUDA components, and dependencies have separate
terms. This repository does not claim that commercial use is permitted.
