# RunPod whisper.cpp worker

RunPod serverless worker that mirrors local `whisper.cpp` transcription path:

```text
server-side ffmpeg -> 16 kHz mono PCM WAV
RunPod worker -> whisper-cli -l en -otxt -nt -mc 0 -sns
```

