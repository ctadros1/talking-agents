"""Validate a student's own narrated recording and create the submission MP4.

Usage: uv run python scripts/finalize_video.py --input ~/Downloads/Christian-Tadros-CAP6938-Narration.mp4
Requires ffmpeg and ffprobe. A silent preview cannot pass as a narrated video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def validate_streams(metadata: dict) -> tuple[str, str]:
    video = next((s for s in metadata["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in metadata["streams"] if s["codec_type"] == "audio"), None)
    if video is None:
        raise ValueError("No video stream: use the slide recording, not an audio-only file")
    if audio is None:
        raise ValueError("No audio stream: the required video must contain your narration")
    return video["codec_name"], audio["codec_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dist/submission/Christian-Tadros-CAP6938-Video.mp4",
    )
    args = parser.parse_args()
    source, out = args.input.expanduser().resolve(), args.out.expanduser().resolve()
    if out.exists():
        raise ValueError(f"Output exists; choose a fresh name: {out}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    video_codec, audio_codec = validate_streams(metadata)
    volume = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(source), "-vn", "-af", "volumedetect", "-f", "null", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"max_volume: ([-\w.]+) dB", volume.stderr)
    if not match or match[1] == "-inf" or float(match[1]) <= -80:
        raise ValueError("The audio is silent; record your narration with the microphone enabled")
    out.parent.mkdir(parents=True, exist_ok=True)
    codecs = ["-c:v", "copy"] if video_codec == "h264" else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    codecs += ["-c:a", "copy"] if audio_codec == "aac" else ["-c:a", "aac", "-b:a", "160k"]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            *codecs,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "null", "-"], check=True)
    receipt = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "source_duration_seconds": metadata["format"].get("duration"),
        "source_video_codec": video_codec,
        "source_audio_codec": audio_codec,
        "max_audio_db": float(match[1]),
        "full_decode_passed": True,
        "student_voice_and_content_review_required": True,
    }
    out.with_suffix(".validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Narrated video validated and saved: {out}")
    print("Play the complete file to confirm your voice, slide coverage and explanation before uploading.")


if __name__ == "__main__":
    main()
