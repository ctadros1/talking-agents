import hashlib
import re
from pathlib import Path

VIEWER = Path(__file__).parents[1] / "viewer"
THREE_SHA256 = "170c6789f43217c96b3170f4b42fafe135de7f7cd48497a4218f9757ee1d49fa"


def test_v2_viewer_loads_only_local_scripts():
    html = (VIEWER / "v2.html").read_text()
    assert "http://" not in html and "https://" not in html
    assert re.findall(r'<script src="([^"]+)"', html) == [
        "vendor/three.min.js",
        "data/replays_v2.js",
        "v2.js",
    ]
    assert (VIEWER / "v2.js").exists() and (VIEWER / "v2.css").exists()


def test_vendored_three_is_pinned_and_licensed():
    assert hashlib.sha256((VIEWER / "vendor" / "three.min.js").read_bytes()).hexdigest() == THREE_SHA256
    assert "The MIT License" in (VIEWER / "vendor" / "three.LICENSE").read_text()


def test_development_replay_is_labeled():
    html = (VIEWER / "v2.html").read_text()
    js = (VIEWER / "v2.js").read_text()
    assert "development-banner" in html
    assert "Scripted reference development data — not learned results" in html
    assert "data.development" in js
