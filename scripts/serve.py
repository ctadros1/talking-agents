"""Build and serve Talking Agents locally. No install or account required."""

from __future__ import annotations

import argparse
import http.server
import webbrowser
from functools import partial

from build_site import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open your default browser")
    args = parser.parse_args()
    root = build()
    url = f"http://localhost:{args.port}/"
    with http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    ) as server:
        print(f"Talking Agents is ready: {url}\nPress Ctrl+C to stop.", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
