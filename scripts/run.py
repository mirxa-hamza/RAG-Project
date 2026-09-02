"""
Start the server and open the browser only once it is actually answering.

    python scripts/run.py                # http://127.0.0.1:8000
    python scripts/run.py --port 9000
    python scripts/run.py --reload       # for development
    python scripts/run.py --no-browser   # start it, open the page yourself

Why this exists: uvicorn imports the application before it binds the port, so for the
first few seconds of a cold start the browser gets ERR_CONNECTION_REFUSED - which looks
like a broken app rather than one that is still starting. This waits for /health to answer
and opens the page then, so the first thing you ever see is the app's own loading screen.

Plain `uvicorn src.main:app --port 8000` still works exactly as before; this is a
convenience wrapper, not a required entry point.
"""
import argparse
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

# Run from anywhere: make sure the project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _open_when_ready(url: str, health_url: str, timeout: float = 300.0) -> None:
    """Polls /health until the server answers, then opens the browser once."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    print(f"\nServer is up - opening {url}\n")
                    webbrowser.open(url)
                    return
        except (urllib.error.URLError, OSError):
            pass  # not listening yet; this is the normal case for the first seconds
        time.sleep(0.5)
    print(f"\nServer did not answer within {timeout:.0f}s - open {url} manually.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Marginalia server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="restart on code changes (development only - it also "
                             "interrupts any ingestion that is still running)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    args = parser.parse_args()

    # 0.0.0.0 means "listen on every interface", which is not an address to browse to.
    browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browse_host}:{args.port}"

    if not args.no_browser:
        threading.Thread(
            target=_open_when_ready,
            args=(url, f"{url}/health"),
            name="open-browser",
            daemon=True,
        ).start()

    print(f"Starting the server on {url} - the browser opens by itself once it responds.")

    import uvicorn

    uvicorn.run("src.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
