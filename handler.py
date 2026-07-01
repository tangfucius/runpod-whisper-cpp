import base64
import contextlib
import json
import os
import resource
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator
from urllib.parse import urlparse

import requests
import runpod


WORKER_STARTED_AT = time.monotonic()
HEARTBEAT_SECONDS = int(os.environ.get("WHISPER_CPP_HEARTBEAT_SECONDS", "10"))
STALL_TIMEOUT_SECONDS = int(os.environ.get("WHISPER_CPP_STALL_TIMEOUT_SECONDS", "120"))


class WhisperStallTimeout(Exception):
    def __init__(self, elapsed_s: float, stdout: str, stderr: str) -> None:
        super().__init__(f"whisper.cpp appeared stalled after {elapsed_s:.3f}s")
        self.elapsed_s = elapsed_s
        self.stdout = stdout
        self.stderr = stderr


def _emit(event: str, **fields: Any) -> None:
    record = {
        "event": event,
        "ts": round(time.time(), 3),
        "worker_uptime_s": round(time.monotonic() - WORKER_STARTED_AT, 3),
        **fields,
    }
    parts = [f"{key}={_log_value(record[key])}" for key in sorted(record)]
    print(" ".join(parts), flush=True)


def _log_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        value = json.dumps(value, default=str, sort_keys=True)
    return shlex.quote(str(value))


def _progress(job: Dict[str, Any], **fields: Any) -> None:
    try:
        runpod.serverless.progress_update(job, fields)
    except Exception as exc:
        _emit("progress_update_failed", error=repr(exc), job_id=job.get("id"))


def _gpu_snapshot() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return {"gpu_available": False, "gpu_error": repr(exc)}

    if result.returncode != 0:
        return {"gpu_available": False, "gpu_error": result.stderr[-500:].strip()}

    first_gpu = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in first_gpu.split(",")]
    if len(parts) != 4:
        return {"gpu_available": False, "gpu_error": f"unexpected nvidia-smi output: {first_gpu}"}

    return {
        "gpu_available": True,
        "gpu_name": parts[0],
        "gpu_util_pct": int(parts[1]),
        "gpu_mem_used_mb": int(parts[2]),
        "gpu_mem_total_mb": int(parts[3]),
    }


def _resource_snapshot() -> Dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_max_rss_mb": round(usage.ru_maxrss / 1024),
        "loadavg_1m": round(os.getloadavg()[0], 3),
        **_gpu_snapshot(),
    }


def _child_process_snapshot(pid: int) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"child_pid": pid}
    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                snapshot["child_state"] = line.split(":", 1)[1].strip()
            elif line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    snapshot["child_rss_mb"] = round(int(parts[1]) / 1024)
            elif line.startswith("voluntary_ctxt_switches:"):
                snapshot["child_voluntary_ctxt_switches"] = int(line.rsplit(None, 1)[1])
            elif line.startswith("nonvoluntary_ctxt_switches:"):
                snapshot["child_nonvoluntary_ctxt_switches"] = int(line.rsplit(None, 1)[1])
    except Exception as exc:
        snapshot["child_process_error"] = repr(exc)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            matching_gpu_processes = 0
            matching_gpu_memory_mb = 0
            matching_gpu_process_name = None
            for line in result.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",", 2)]
                if len(parts) == 3 and parts[0] == str(pid):
                    matching_gpu_processes += 1
                    matching_gpu_process_name = parts[1]
                    matching_gpu_memory_mb += int(parts[2])
            snapshot["child_gpu_process_count"] = matching_gpu_processes
            snapshot["child_gpu_process_name"] = matching_gpu_process_name
            snapshot["child_gpu_used_memory_mb"] = matching_gpu_memory_mb
        else:
            snapshot["child_gpu_process_error"] = result.stderr[-500:].strip()
    except Exception as exc:
        snapshot["child_gpu_process_error"] = repr(exc)

    return snapshot


@contextlib.contextmanager
def _stage(job: Dict[str, Any], name: str, **fields: Any) -> Iterator[None]:
    start = time.monotonic()
    job_id = job.get("id")
    _emit("stage_start", stage=name, job_id=job_id, **fields, **_resource_snapshot())
    _progress(job, stage=name, status="start")
    try:
        yield
    except Exception as exc:
        elapsed = round(time.monotonic() - start, 3)
        _emit(
            "stage_error",
            stage=name,
            job_id=job_id,
            elapsed_s=elapsed,
            error=repr(exc),
            **_resource_snapshot(),
        )
        _progress(job, stage=name, status="error", elapsed_s=elapsed, error=str(exc))
        raise
    else:
        elapsed = round(time.monotonic() - start, 3)
        _emit("stage_end", stage=name, job_id=job_id, elapsed_s=elapsed, **_resource_snapshot())
        _progress(job, stage=name, status="end", elapsed_s=elapsed)


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _download_audio(url: str, dest: Path) -> int:
    # Do not log presigned URLs; they may grant temporary access to private audio.
    total_bytes = 0
    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    total_bytes += len(chunk)
                    f.write(chunk)
    return total_bytes


def _write_base64_audio(data: str, dest: Path) -> int:
    audio_bytes = base64.b64decode(data)
    dest.write_bytes(audio_bytes)
    return len(audio_bytes)


def _kill_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run(command: list[str], timeout: int, job: Dict[str, Any]) -> subprocess.CompletedProcess:
    start = time.monotonic()
    with tempfile.TemporaryFile("w+") as stdout_file, tempfile.TemporaryFile("w+") as stderr_file:
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                process.kill()
                process.wait()
                stdout_file.seek(0)
                stderr_file.seek(0)
                raise subprocess.TimeoutExpired(
                    command,
                    timeout,
                    output=stdout_file.read(),
                    stderr=stderr_file.read(),
                )

            wait_seconds = min(HEARTBEAT_SECONDS, max(1, timeout - elapsed))
            try:
                process.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                heartbeat_elapsed = round(time.monotonic() - start, 3)
                _emit(
                    "transcribe_running",
                    job_id=job.get("id"),
                    stage="transcribe",
                    elapsed_s=heartbeat_elapsed,
                    **_child_process_snapshot(process.pid),
                    **_resource_snapshot(),
                )

                if STALL_TIMEOUT_SECONDS > 0 and heartbeat_elapsed >= STALL_TIMEOUT_SECONDS:
                    _kill_process(process)
                    stdout_file.seek(0)
                    stderr_file.seek(0)
                    raise WhisperStallTimeout(
                        heartbeat_elapsed,
                        stdout_file.read(),
                        stderr_file.read(),
                    )

        stdout_file.seek(0)
        stderr_file.seek(0)
        return subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout_file.read(),
            stderr_file.read(),
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
    job_started_at = time.monotonic()
    job_input = job.get("input") or {}
    job_id = job.get("id")
    timings: Dict[str, float] = {}
    _emit(
        "job_received",
        job_id=job_id,
        input_keys=sorted(job_input.keys()),
        has_audio=bool(job_input.get("audio")),
        has_audio_base64=bool(job_input.get("audio_base64")),
        **_resource_snapshot(),
    )

    if bool(job_input.get("audio")) == bool(job_input.get("audio_base64")):
        _emit("job_rejected", job_id=job_id, reason="invalid_audio_input")
        return {"error": "Provide exactly one of audio or audio_base64"}

    model_name = str(job_input.get("model") or os.environ.get("WHISPER_CPP_MODEL") or "large-v3")
    transcription_format = str(job_input.get("transcription") or "plain_text")
    language = str(job_input.get("language") or "en")
    timeout = int(os.environ.get("WHISPER_CPP_TIMEOUT_SECONDS", "3600"))
    threads = str(job_input.get("threads") or os.environ.get("WHISPER_CPP_THREADS", "4"))

    # Keep these defaults aligned with the expected local whisper.cpp invocation:
    #   -nt    no timestamps
    #   -mc 0  no previous text context
    #   -sns   suppress non-speech tokens
    no_timestamps = _bool(job_input.get("no_timestamps"), True)
    max_context = str(job_input.get("max_context", "0"))
    suppress_non_speech = _bool(job_input.get("suppress_non_speech"), True)

    whisper_bin = Path(os.environ.get("WHISPER_CPP_BIN", "/opt/whisper.cpp/build/bin/whisper-cli"))
    if not whisper_bin.exists():
        _emit("job_rejected", job_id=job_id, reason="missing_whisper_cli", whisper_bin=str(whisper_bin))
        return {"error": f"whisper-cli not found at {whisper_bin}"}

    try:
        with _stage(job, "resolve_model", model=model_name):
            model_path = _model_path(model_name)
    except Exception as exc:
        _emit("job_rejected", job_id=job_id, reason="missing_model", model=model_name, error=repr(exc))
        return {"error": str(exc)}

    work_dir = Path(tempfile.mkdtemp(prefix="whisper-cpp-"))
    try:
        # Audio should be pre-converted to 16 kHz mono PCM WAV before staging it.
        # The worker only downloads and transcribes, avoiding runtime ffmpeg
        # overhead inside RunPod.
        wav_path = work_dir / "audio.wav"
        out_base = work_dir / "transcript"
        out_txt = work_dir / "transcript.txt"

        if job_input.get("audio"):
            with _stage(job, "download_audio"):
                stage_started_at = time.monotonic()
                audio_bytes = _download_audio(str(job_input["audio"]), wav_path)
                timings["download_audio_s"] = round(time.monotonic() - stage_started_at, 3)
                _emit(
                    "audio_downloaded",
                    job_id=job_id,
                    bytes=audio_bytes,
                    parsed_url_host=urlparse(str(job_input["audio"])).hostname,
                )
        else:
            with _stage(job, "decode_base64_audio"):
                stage_started_at = time.monotonic()
                audio_bytes = _write_base64_audio(str(job_input["audio_base64"]), wav_path)
                timings["decode_base64_audio_s"] = round(time.monotonic() - stage_started_at, 3)
                _emit("audio_decoded", job_id=job_id, bytes=audio_bytes)

        wav_size = wav_path.stat().st_size
        _emit("audio_ready", job_id=job_id, wav_size_bytes=wav_size)

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

        replacements = {
            str(whisper_bin): "<whisper-cli>",
            str(model_path): "<model>",
            str(wav_path): "<audio>",
        }
        safe_command = [replacements.get(arg, arg) for arg in command]
        _emit(
            "whisper_command_ready",
            job_id=job_id,
            command=safe_command,
            stall_timeout_s=STALL_TIMEOUT_SECONDS,
            timeout_s=timeout,
            **_resource_snapshot(),
        )

        with _stage(job, "transcribe", model=model_name, threads=threads, timeout_s=timeout):
            stage_started_at = time.monotonic()
            result = _run(command, timeout=timeout, job=job)
            timings["transcribe_s"] = round(time.monotonic() - stage_started_at, 3)

        if result.returncode != 0:
            _emit(
                "whisper_failed",
                job_id=job_id,
                returncode=result.returncode,
                stderr_tail=result.stderr[-4000:],
                stdout_tail=result.stdout[-4000:],
                **_resource_snapshot(),
            )
            return {
                "error": "whisper.cpp transcription failed",
                "stderr": result.stderr[-4000:],
                "stdout": result.stdout[-4000:],
            }
        if not out_txt.exists():
            _emit(
                "transcript_missing",
                job_id=job_id,
                stderr_tail=result.stderr[-4000:],
                stdout_tail=result.stdout[-4000:],
                **_resource_snapshot(),
            )
            return {
                "error": "whisper.cpp did not produce transcript.txt",
                "stderr": result.stderr[-4000:],
                "stdout": result.stdout[-4000:],
            }

        with _stage(job, "read_transcript"):
            stage_started_at = time.monotonic()
            text = out_txt.read_text(encoding="utf-8").strip()
            timings["read_transcript_s"] = round(time.monotonic() - stage_started_at, 3)

        total_elapsed_s = round(time.monotonic() - job_started_at, 3)
        _emit(
            "job_done",
            job_id=job_id,
            total_elapsed_s=total_elapsed_s,
            transcript_chars=len(text),
            **_resource_snapshot(),
        )
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
            "timings": {
                **timings,
                "total_s": total_elapsed_s,
            },
        }
    except subprocess.TimeoutExpired as exc:
        _emit(
            "job_timeout",
            job_id=job_id,
            timeout_s=exc.timeout,
            stdout_tail=(exc.output or "")[-4000:],
            stderr_tail=(exc.stderr or "")[-4000:],
            **_resource_snapshot(),
        )
        return {
            "error": "whisper.cpp transcription timed out",
            "timeout_seconds": exc.timeout,
            "stdout": (exc.output or "")[-4000:],
            "stderr": (exc.stderr or "")[-4000:],
            "timings": timings,
        }
    except WhisperStallTimeout as exc:
        _emit(
            "whisper_stalled",
            job_id=job_id,
            stall_timeout_s=STALL_TIMEOUT_SECONDS,
            elapsed_s=round(exc.elapsed_s, 3),
            stdout_tail=exc.stdout[-4000:],
            stderr_tail=exc.stderr[-4000:],
            **_resource_snapshot(),
        )
        return {
            "error": "whisper.cpp transcription appeared stalled",
            "stall_timeout_seconds": STALL_TIMEOUT_SECONDS,
            "elapsed_seconds": round(exc.elapsed_s, 3),
            "stdout": exc.stdout[-4000:],
            "stderr": exc.stderr[-4000:],
            "timings": timings,
        }
    except Exception as exc:
        _emit("job_error", job_id=job_id, error=repr(exc), **_resource_snapshot())
        return {"error": str(exc)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _emit("work_dir_removed", job_id=job_id)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
