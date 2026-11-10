import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("finalize_video", Path(__file__).parents[1] / "scripts/finalize_video.py")
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


def test_silent_export_cannot_be_accepted_as_the_required_narrated_video():
    with pytest.raises(ValueError, match="audio"):
        video.validate_streams({"streams": [{"codec_type": "video", "codec_name": "h264"}]})


def test_audio_only_file_cannot_be_accepted_as_a_system_video():
    with pytest.raises(ValueError, match="video"):
        video.validate_streams({"streams": [{"codec_type": "audio", "codec_name": "aac"}]})


def test_recording_requires_both_video_and_audio():
    streams = {"streams": [{"codec_type": "video", "codec_name": "vp9"}, {"codec_type": "audio", "codec_name": "opus"}]}
    assert video.validate_streams(streams) == ("vp9", "opus")
