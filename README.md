# Arteri RunPod whisper.cpp worker

RunPod serverless worker that mirrors Arteri's local `whisper.cpp` transcription path:

```text
server-side ffmpeg in Arteri -> 16 kHz mono PCM WAV
RunPod worker -> whisper-cli -l en -otxt -nt -mc 0 -sns
```

This is intended as a parity-first replacement for generic faster-whisper workers when
clinic audio quality/stability matters more than using faster-whisper defaults.

## Input

Compatible with Arteri's RunPod URL staging flow. The `audio` URL should point to a pre-converted 16 kHz mono PCM WAV file:

```json
{
  "input": {
    "audio": "https://signed-r2-url.example/audio.mp3",
    "model": "large-v3",
    "transcription": "plain_text",
    "language": "en"
  }
}
```

Optional overrides:

```json
{
  "input": {
    "no_timestamps": true,
    "max_context": "0",
    "suppress_non_speech": true,
    "threads": 4
  }
}
```

## Output

```json
{
  "transcription": "...",
  "segments": [],
  "detected_language": "en",
  "model": "large-v3",
  "device": "cuda",
  "engine": "whisper.cpp"
}
```

## Build notes

The Docker image clones `ggml-org/whisper.cpp` at a pinned commit and downloads
`ggml-large-v3.bin` during image build. This makes cold starts larger but keeps runtime
behavior predictable. Runtime audio conversion is intentionally not included; Arteri does
conversion before uploading to R2 so RunPod time is spent on transcription only.
