# RunPod whisper.cpp worker

RunPod serverless worker that mirrors a local [whisper.cpp](https://github.com/ggml-org/whisper.cpp) transcription path:

```text
server-side ffmpeg -> 16 kHz mono PCM WAV
RunPod worker -> whisper-cli -l en -otxt -nt -mc 0 -sns
```

The worker expects either:

- `audio`: a URL to a staged 16 kHz mono PCM WAV file, such as a presigned R2 URL
- `audio_base64`: base64-encoded 16 kHz mono PCM WAV bytes

Pre-converting and staging audio keeps ffmpeg work out of the RunPod worker.

## Prepare audio and upload to a Cloudflare R2 staging bucket

Set common environment variables:

```bash
export R2_ACCOUNT_ID="<cloudflare-account-id>"
export R2_BUCKET="<staging-bucket>"
export R2_ACCESS_KEY_ID="<r2-access-key-id>"
export R2_SECRET_ACCESS_KEY="<r2-secret-access-key>"
export R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
```

### CLI

Requires `ffmpeg` and `awscli`.

```bash
INPUT="input.mp3"
OUTPUT="audio.wav"
KEY="staging/$(uuidgen).wav"

# Convert to 16 kHz mono signed 16-bit PCM WAV.
ffmpeg -y -i "$INPUT" -ac 1 -ar 16000 -c:a pcm_s16le "$OUTPUT"

# Upload to R2 using the S3-compatible API.
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
aws s3 cp "$OUTPUT" "s3://${R2_BUCKET}/${KEY}" \
  --endpoint-url "$R2_ENDPOINT" \
  --content-type audio/wav
```

### Python

Requires `ffmpeg` on `PATH` and `boto3`.

```python
import os
import subprocess
import uuid

import boto3

input_path = "input.mp3"
wav_path = "audio.wav"
key = f"staging/{uuid.uuid4()}.wav"

subprocess.run([
    "ffmpeg", "-y",
    "-i", input_path,
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    wav_path,
], check=True)

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)

with open(wav_path, "rb") as f:
    s3.upload_fileobj(
        f,
        os.environ["R2_BUCKET"],
        key,
        ExtraArgs={"ContentType": "audio/wav"},
    )

print(key)
```

### TypeScript

Requires `ffmpeg` on `PATH` and `@aws-sdk/client-s3`.

```ts
import { createReadStream } from "node:fs";
import { spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";

const inputPath = "input.mp3";
const wavPath = "audio.wav";
const key = `staging/${randomUUID()}.wav`;

const ffmpeg = spawnSync("ffmpeg", [
  "-y",
  "-i", inputPath,
  "-ac", "1",
  "-ar", "16000",
  "-c:a", "pcm_s16le",
  wavPath,
], { stdio: "inherit" });

if (ffmpeg.status !== 0) {
  throw new Error(`ffmpeg failed with exit code ${ffmpeg.status}`);
}

const s3 = new S3Client({
  region: "auto",
  endpoint: process.env.R2_ENDPOINT!,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID!,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
  },
});

await s3.send(new PutObjectCommand({
  Bucket: process.env.R2_BUCKET!,
  Key: key,
  Body: createReadStream(wavPath),
  ContentType: "audio/wav",
}));

console.log(key);
```

## Run the RunPod endpoint with the staged R2 URL

Create a presigned URL for the staged object, then pass it as `input.audio` to the RunPod endpoint:

```bash
export RUNPOD_API_KEY="<runpod-api-key>"
export RUNPOD_ENDPOINT_ID="<runpod-endpoint-id>"

R2_URL=$(AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
aws s3 presign "s3://${R2_BUCKET}/${KEY}" \
  --endpoint-url "$R2_ENDPOINT" \
  --expires-in 3600)

curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg audio "$R2_URL" '{input: {audio: $audio, model: "large-v3", language: "en"}}')"
```

## Runtime instrumentation

The worker emits newline-delimited JSON logs to stdout. RunPod captures these in
endpoint logs, and the handler also sends compact JSON progress updates while a
job is running.

Useful events when debugging timeouts:

- `job_received`: handler accepted the job and records input shape.
- `stage_start` / `stage_end`: records elapsed time for `resolve_model`,
  `download_audio` or `decode_base64_audio`, `transcribe`, and `read_transcript`.
- `audio_downloaded`: records staged audio byte count and URL host without
  logging the full presigned URL.
- `whisper_command_ready`: records sanitized whisper.cpp args and resource state.
- `whisper_heartbeat`: emitted every `WHISPER_CPP_HEARTBEAT_SECONDS` while
  `whisper-cli` is still running.
- `job_timeout`: emitted if the whisper.cpp subprocess exceeds
  `WHISPER_CPP_TIMEOUT_SECONDS`.

Each log includes process RSS, 1-minute load average, and GPU utilization/memory
from `nvidia-smi` when available. Set `WHISPER_CPP_HEARTBEAT_SECONDS` to adjust
heartbeat frequency; the default is 10 seconds.
