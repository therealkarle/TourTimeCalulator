"""Obtain a Strava refresh token through OAuth and store it in .env.

The user must approve the application in a browser once. The local callback
server then exchanges the authorization code for a refresh token.
"""

from __future__ import annotations

import argparse
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"


def read_env_value(name: str) -> Optional[str]:
    """Read a simple KEY=value entry without printing secrets."""
    if not ENV_PATH.exists():
        return None
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def update_env_value(name: str, value: str) -> None:
    """Update one .env value while preserving the rest of the file."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True) if ENV_PATH.exists() else []
    prefix = f"{name}="
    for index, line in enumerate(lines):
        if line.lstrip().startswith(prefix):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f"{prefix}{value}{newline}"
            break
    else:
        if lines and not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(f"{prefix}{value}\n")

    ENV_PATH.write_text("".join(lines), encoding="utf-8")


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Capture the one OAuth callback and return a small browser response."""

    result: dict[str, str] = {}
    expected_state = ""

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        query = parse_qs(urlparse(self.path).query)
        OAuthCallbackHandler.result = {
            key: values[0] for key, values in query.items() if values
        }
        valid = OAuthCallbackHandler.result.get("state") == self.expected_state
        if not valid:
            OAuthCallbackHandler.result = {"error": "Invalid OAuth state"}

        body = (
            "<html><body><h2>Strava authorization received.</h2>"
            "<p>You can close this window and return to the terminal.</p></body></html>"
        ).encode("utf-8")
        self.send_response(200 if valid else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def exchange_code(client_id: str, client_secret: str, code: str) -> str:
    payload = urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
        }
    ).encode("ascii")
    request = Request(TOKEN_URL, data=payload, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urlopen(request, timeout=30) as response:
            import json

            data = json.load(response)
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Strava token exchange failed: {error}") from error

    refresh_token = data.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("Strava response did not contain a refresh_token")
    return refresh_token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Local callback port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL instead of opening it")
    args = parser.parse_args()

    client_id = os.getenv("STRAVA_CLIENT_ID") or read_env_value("STRAVA_CLIENT_ID")
    client_secret = os.getenv("STRAVA_CLIENT_SECRET") or read_env_value("STRAVA_CLIENT_SECRET")
    if not client_id or not client_secret or client_id.startswith("your_") or client_secret.startswith("your_"):
        raise SystemExit("STRAVA_CLIENT_ID und STRAVA_CLIENT_SECRET müssen in .env gesetzt sein.")

    state = secrets.token_urlsafe(24)
    redirect_uri = f"http://localhost:{args.port}/callback"
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "approval_prompt": "force",
        "scope": "read,activity:read_all",
        "state": state,
    }
    authorization_url = f"{AUTH_URL}?{urlencode(params)}"

    handler = OAuthCallbackHandler
    handler.expected_state = state
    handler.result = {}
    try:
        server = HTTPServer(("localhost", args.port), handler)
    except OSError as error:
        raise SystemExit(f"Lokaler Callback-Port {args.port} konnte nicht geöffnet werden: {error}") from error

    print("Öffne Strava zur Autorisierung …")
    if args.no_browser:
        print(authorization_url)
    elif not webbrowser.open(authorization_url):
        print(f"Browser konnte nicht geöffnet werden. Öffne diese URL manuell:\n{authorization_url}")

    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()
    server_thread.join(timeout=300)
    server.server_close()
    if not handler.result:
        raise SystemExit("Kein OAuth-Callback innerhalb von 5 Minuten erhalten.")
    if handler.result.get("error"):
        raise SystemExit(f"Strava-Autorisierung fehlgeschlagen: {handler.result['error']}")

    refresh_token = exchange_code(client_id, client_secret, handler.result["code"])
    update_env_value("STRAVA_REFRESH_TOKEN", refresh_token)
    print(f"Refresh-Token wurde sicher in {ENV_PATH} gespeichert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
