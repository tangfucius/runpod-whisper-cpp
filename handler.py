import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import requests
import runpod


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _download_audio(url: str, dest: Path) -> None:
    # Do not log presigned URLs; they may grant temporary access to private audio.
    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _write_base64_audio(data: str, dest: Path) -> None:
    dest.write_bytes(base64.b64decode(data))


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def _model_path(model_name: str) -> Path:
    model_dir = Path(os.environ.get("WHISPER_CPP_MODEL_DIR", "/opt/whisper.cpp/models"))
    path = model_dir / f"ggml-{model_name}.bin"
    if not path.exists():
        raise ValueError(f"Unsupported or unavailable model: {model_name}")
    return path


def _format_output(text: str, format_type: str) -> str:
    # whisper.cpp .txt already contains readable text with newlines. Keep it as-is for both
    # plain_text and formatted_text so downstream note generation sees local parity output.
    return text.strip()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input") or {}
    if bool(job_input.get("audio")) == bool(job_input.get("audio_base64")):
        return {"error": "Provide exactly one of audio or audio_base64"}

    model_name = str(job_input.get("model") or os.environ.get("WHISPER_CPP_MODEL") or "large-v3")
    transcription_format = str(job_input.get("transcription") or "plain_text")
    language = str(job_input.get("language") or "en")
    timeout = int(os.environ.get("WHISPER_CPP_TIMEOUT_SECONDS", "3600"))
    threads = str(job_input.get("threads") or os.environ.get("WHISPER_CPP_THREADS", "4"))

    # Keep these defaults aligned with Arteri's local whisper.cpp invocation:
    #   -nt    no timestamps
    #   -mc 0  no previous text context
    #   -sns   suppress non-speech tokens
    no_timestamps = _bool(job_input.get("no_timestamps"), True)
    max_context = str(job_input.get("max_context", "0"))
    suppress_non_speech = _bool(job_input.get("suppress_non_speech"), True)

    whisper_bin = Path(os.environ.get("WHISPER_CPP_BIN", "/opt/whisper.cpp/build/bin/whisper-cli"))
    if not whisper_bin.exists():
        return {"error": f"whisper-cli not found at {whisper_bin}"}

    try:
        model_path = _model_path(model_name)
    except Exception as exc:
        return {"error": str(exc)}

    work_dir = Path(tempfile.mkdtemp(prefix="arteri-whisper-cpp-"))
    try:
        # Arteri pre-converts audio server-side to 16 kHz mono PCM WAV before
        # staging it in R2. The worker only downloads and transcribes, avoiding
        # runtime ffmpeg overhead inside RunPod.
        wav_path = work_dir / "audio.wav"
        out_base = work_dir / "transcript"
        out_txt = work_dir / "transcript.txt"

        if job_input.get("audio"):
            _download_audio(str(job_input["audio"]), wav_path)
        else:
            _write_base64_audio(str(job_input["audio_base64"]), wav_path)

        command = [
            str(whisper_bin),
            "-m",
            str(model_path),
            "-f",
            str(wav_path),
            "-l",
            language,
            "-otxt",
            "-of",
            str(out_base),
            "-t",
            threads,
            "-mc",
            max_context,
        ]
        if no_timestamps:
            command.append("-nt")
        if suppress_non_speech:
            command.append("-sns")

        result = _run(command, timeout=timeout)
        if result.returncode != 0:
            return {
                "error": "whisper.cpp transcription failed",
                "stderr": result.stderr[-4000:],
                "stdout": result.stdout[-4000:],
            }
        if not out_txt.exists():
            return {
                "error": "whisper.cpp did not produce transcript.txt",
                "stderr": result.stderr[-4000:],
                "stdout": result.stdout[-4000:],
            }

        text = out_txt.read_text(encoding="utf-8").strip()
        return {
            "transcription": _format_output(text, transcription_format),
            "segments": [],
            "detected_language": language,
            "model": model_name,
            "device": "cuda",
            "engine": "whisper.cpp",
            "options": {
                "no_timestamps": no_timestamps,
                "max_context": max_context,
                "suppress_non_speech": suppress_non_speech,
                "threads": int(threads),
            },
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
