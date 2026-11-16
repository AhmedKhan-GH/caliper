"""The sidecar's client of Caliper API, on loopback only.

A tool runs by calling Caliper API (contracts sections 8 and 15):

- `app.info`, `app.applets`, `app.launch` and `capture.frame` use the API v0
  endpoints that exist today (`GET /v0/info`, `GET /v0/applets`,
  `POST /v0/applets/{id}/launch`, `GET /v0/capture.png`);
- every other command is `POST /v0/commands/{name}` with the arguments as the
  JSON body, `X-Caliper-Origin: agent` and `X-Caliper-Group: <the turn's group>`,
  so that the edits of one prompt undo as one step, and in a widget session
  `X-Caliper-Preview: <the session's preview>`, so that its edits stack on the
  preview the app opened and its reads see it. A 404 means the app has no
  command layer yet.

The address and the session token come from the API file: `--api-file`, else
`api.json` in `CALIPER_DATA_DIR`, else in the app-data folder of the OS. The
file is read when a tool first needs it and again after a refused token (the
app restarted). It is hostile input: its size is bounded, the token must look
like one, and an address that is not loopback is refused, so the token is
never sent off the machine. The token is registered with the scrubber and is
never logged.

A call may wait while the person decides in the app (up to 120 seconds), so the
read timeout is 130 seconds. A PNG result (a capture, or `png_base64` in a
command's result) comes back as an image for the model.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx2

from .keyguard import SCRUBBER, describe_exception

MAX_API_FILE_BYTES = 64 << 10
MAX_ARGS_BYTES = 1 << 20
MAX_RESPONSE_BYTES = 32 << 20
# A PNG larger than this is not given to the model. Its base64 text then stays
# under 5 MB, the size limit of one image in a Claude API request.
MAX_IMAGE_BYTES = (5 * 1000 * 1000 * 3) // 4
MAX_RESULT_TEXT = 96 << 10
CALL_TIMEOUT_S = 130.0
CONNECT_TIMEOUT_S = 5.0

_TOKEN = re.compile(r"^[A-Za-z0-9_\-]{16,256}$")
_APPLET_ID = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ApiFileError(Exception):
    """The API file is missing or unusable, or a response broke a limit. The
    message is safe to show."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    token: str = field(repr=False)

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{host}:{self.port}"


@dataclass(frozen=True)
class RawResponse:
    status_code: int
    content: bytes


@dataclass
class ToolOutput:
    """What a call gave: a result or an error, with an optional PNG."""

    ok: bool
    text: str
    result: Any = None
    error_code: str = ""
    png: bytes | None = None
    width: int = 0
    height: int = 0


def default_api_file(environ: dict | None = None) -> Path:
    env = os.environ if environ is None else environ
    data_dir = env.get("CALIPER_DATA_DIR", "")
    if data_dir:
        return Path(data_dir) / "api.json"
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Caliper" / "api.json"
    if sys.platform.startswith("win"):
        appdata = env.get("APPDATA", "")
        return (Path(appdata) if appdata else Path("caliper-data")) / "Caliper" / "api.json"
    xdg = env.get("XDG_DATA_HOME", "")
    return (Path(xdg) / "caliper" if xdg else home / ".local" / "share" / "caliper") / "api.json"


def parse_api_file(data: bytes) -> Endpoint:
    if len(data) > MAX_API_FILE_BYTES:
        raise ApiFileError("caliper.bad_api_file", "The API file is too large.")
    try:
        doc = json.loads(data)
    except ValueError:
        raise ApiFileError("caliper.bad_api_file", "The API file is not JSON.") from None
    if not isinstance(doc, dict):
        raise ApiFileError("caliper.bad_api_file", "The API file is not a JSON object.")
    token = doc.get("token")
    if not isinstance(token, str) or not _TOKEN.match(token):
        raise ApiFileError("caliper.bad_api_file", "The API file has no usable session token.")
    port = doc.get("port")
    url = doc.get("url")
    host = "127.0.0.1"
    if isinstance(url, str) and url:
        parts = urlsplit(url)
        if parts.scheme != "http" or parts.hostname is None:
            raise ApiFileError("caliper.bad_api_file", "The API file's url is not an http address.")
        host = parts.hostname
        if parts.port is not None:
            if isinstance(port, int) and port != parts.port:
                raise ApiFileError("caliper.bad_api_file", "The API file's url and port disagree.")
            port = parts.port
    if host not in _LOOPBACK_HOSTS:
        raise ApiFileError("caliper.not_loopback",
                           "The API file names an address that is not on this machine; the sidecar refuses it.")
    if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
        raise ApiFileError("caliper.bad_api_file", "The API file has no usable port.")
    return Endpoint(host=host.strip("[]"), port=port, token=token)


def read_api_file(path: Path) -> Endpoint:
    try:
        with path.open("rb") as f:
            data = f.read(MAX_API_FILE_BYTES + 1)
    except FileNotFoundError:
        raise ApiFileError("caliper.no_api_file",
                           "Caliper API is not running: there is no API file. Start Caliper with the API on.") from None
    except OSError as exc:
        raise ApiFileError("caliper.no_api_file", f"The API file cannot be read: {exc.strerror or 'error'}.") from None
    endpoint = parse_api_file(data)
    SCRUBBER.register(endpoint.token)
    return endpoint


def png_size(data: bytes) -> tuple[int, int] | None:
    """The width and height in a PNG's IHDR chunk, or None if it is not a PNG."""
    if len(data) < 24 or not data.startswith(_PNG_SIGNATURE) or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_TEXT:
        return text
    return text[:MAX_RESULT_TEXT] + f"\n[The result was cut at {MAX_RESULT_TEXT} characters.]"


def _error(code: str, message: str, where: str = "") -> ToolOutput:
    body = {"code": code, "message": message}
    if where:
        body["where"] = where
    return ToolOutput(ok=False, text=json.dumps({"error": body}, ensure_ascii=False), error_code=code)


def _with_image(result: dict, png: bytes) -> ToolOutput:
    size = png_size(png)
    if size is None:
        return _error("caliper.bad_image", "Caliper returned an image that is not a PNG.")
    if len(png) > MAX_IMAGE_BYTES:
        return _error("caliper.image_too_large",
                      f"The image is {len(png)} bytes, more than the {MAX_IMAGE_BYTES} bytes a model request takes. "
                      "Render a smaller panel instead.")
    rest = {k: v for k, v in result.items() if k != "png_base64"}
    rest.setdefault("width", size[0])
    rest.setdefault("height", size[1])
    rest["image"] = "attached as a PNG"
    return ToolOutput(ok=True, text=_truncate(json.dumps(rest, sort_keys=True, ensure_ascii=False)), result=rest,
                      png=png, width=size[0], height=size[1])


def output_from_envelope(envelope: Any) -> ToolOutput:
    """A command result envelope (contracts section 7) as a tool output."""
    if not isinstance(envelope, dict) or not isinstance(envelope.get("ok"), bool):
        return _error("caliper.bad_response", "Caliper API answered with something that is not a result envelope.")
    if not envelope["ok"]:
        err = envelope.get("error") if isinstance(envelope.get("error"), dict) else {}
        code = err.get("code") if isinstance(err.get("code"), str) else "command.failed"
        message = err.get("message") if isinstance(err.get("message"), str) else "The command failed."
        where = err.get("where") if isinstance(err.get("where"), str) else ""
        out = _error(code, message, where)
        if isinstance(err.get("approval"), str):
            out.text = json.dumps({"error": {**json.loads(out.text)["error"], "approval": err["approval"]}})
        return out
    result = envelope.get("result")
    if isinstance(result, dict) and isinstance(result.get("png_base64"), str):
        try:
            png = base64.b64decode(result["png_base64"], validate=True)
        except (binascii.Error, ValueError):
            return _error("caliper.bad_image", "Caliper returned an image that is not valid base64.")
        return _with_image(result, png)
    text = json.dumps(result, sort_keys=True, ensure_ascii=False)
    return ToolOutput(ok=True, text=_truncate(text), result=result)


class CaliperApi:
    """Calls Caliper API on behalf of the model. One instance per sidecar."""

    def __init__(self, api_file: Path | None = None, transport: httpx2.AsyncBaseTransport | None = None, *,
                 not_for_agents: frozenset[str] = frozenset()) -> None:
        self._api_file = api_file
        # The table's commands whose `agents` flag is false (tools.not_for_agents).
        self._not_for_agents = frozenset(not_for_agents)
        self._endpoint: Endpoint | None = None
        self._client = httpx2.AsyncClient(
            transport=transport,
            trust_env=False,   # loopback: never through a proxy
            follow_redirects=False,
            timeout=httpx2.Timeout(CALL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _path(self) -> Path:
        return self._api_file if self._api_file is not None else default_api_file()

    def _load(self, reload: bool = False) -> Endpoint:
        if self._endpoint is None or reload:
            self._endpoint = read_api_file(self._path())
        return self._endpoint

    async def _send(self, method: str, path: str, *, body: bytes | None = None,
                    headers: dict | None = None) -> RawResponse:
        """Sends one request, re-reading the API file once after a refused token."""
        for attempt in (0, 1):
            endpoint = self._load(reload=attempt == 1)
            h = {"Authorization": f"Bearer {endpoint.token}", "Accept": "application/json, image/png"}
            if headers:
                h.update(headers)
            if body is not None:
                h["Content-Type"] = "application/json"
            request = self._client.build_request(method, endpoint.base_url + path, content=body, headers=h)
            response = await self._client.send(request, stream=True)
            try:
                chunks = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        raise ApiFileError("caliper.response_too_large", "Caliper API sent more than 32 MiB.")
                    chunks.append(chunk)
            finally:
                await response.aclose()
            raw = RawResponse(response.status_code, b"".join(chunks))
            if raw.status_code == 401 and attempt == 0:
                continue
            return raw
        return raw  # pragma: no cover

    async def call(self, command: str, args: dict, group: str = "", preview: str = "") -> ToolOutput:
        try:
            return await self._call(command, args, group, preview)
        except ApiFileError as exc:
            return _error(exc.code, str(exc))
        except httpx2.ConnectError:
            return _error("caliper.unreachable",
                          "Caliper API is not reachable. Is Caliper running with the API on?")
        except httpx2.TimeoutException:
            return _error("caliper.timeout",
                          f"Caliper API did not answer within {int(CALL_TIMEOUT_S)} seconds.")
        except httpx2.HTTPError as exc:
            return _error("caliper.transport", f"The call to Caliper API failed: {describe_exception(exc)}")

    async def _call(self, command: str, args: dict, group: str, preview: str = "") -> ToolOutput:
        if command in self._not_for_agents:
            # Offered to no model (tools.offered_to_the_model); a call made anyway is
            # not sent, and is refused with the command layer's own code for it.
            return _error("command.not_allowed", f"{command} is the person's own tool, so an agent cannot run it; to "
                                                 "take back a change, send an edit that reverts it, which the person "
                                                 "reviews.", "origin")
        if command == "app.info":
            return self._plain(await self._send("GET", "/v0/info"))
        if command == "app.applets":
            return self._plain(await self._send("GET", "/v0/applets"))
        if command == "app.launch":
            applet = args.get("id")
            if not isinstance(applet, str) or not _APPLET_ID.match(applet):
                return _error("command.invalid_args", "id must be 1 to 128 characters of A-Z a-z 0-9 . _ -.", "args.id")
            response = await self._send("POST", f"/v0/applets/{quote(applet, safe='')}/launch")
            return self._plain(response, {404: "app.applet_not_found", 409: "app.launch_refused",
                                          400: "command.invalid_args"})
        if command == "capture.frame":
            response = await self._send("GET", "/v0/capture.png")
            if response.status_code == 200:
                return _with_image({}, response.content)
            return self._plain(response, {501: "capture.unavailable", 504: "capture.timeout",
                                          503: "capture.unavailable"})

        body = json.dumps(args, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_ARGS_BYTES:
            return _error("command.invalid_args", "The arguments are larger than 1 MiB, the most Caliper API takes.")
        headers = {"X-Caliper-Origin": "agent"}
        if group:
            headers["X-Caliper-Group"] = group
        if preview:
            headers["X-Caliper-Preview"] = preview
        response = await self._send("POST", f"/v0/commands/{quote(command, safe='.')}", body=body, headers=headers)
        if response.status_code == 404:
            return _error("caliper.no_command_layer",
                          "This Caliper app has no command layer yet (Caliper API answered 404), "
                          "so only app_info, app_applets, app_launch and capture_frame work.")
        if response.status_code != 200:
            return self._status_error(response)
        try:
            envelope = json.loads(response.content)
        except ValueError:
            return _error("caliper.bad_response", "Caliper API answered with something that is not JSON.")
        return output_from_envelope(envelope)

    def _status_error(self, response: RawResponse, codes: dict | None = None) -> ToolOutput:
        reason = ""
        try:
            doc = json.loads(response.content)
            if isinstance(doc, dict) and isinstance(doc.get("error"), str):
                reason = doc["error"][:2000]
        except ValueError:
            pass
        code = (codes or {}).get(response.status_code, f"caliper.http_{response.status_code}")
        message = f"Caliper API answered HTTP {response.status_code}"
        return _error(code, f"{message}: {reason}" if reason else f"{message}.")

    def _plain(self, response: RawResponse, codes: dict | None = None) -> ToolOutput:
        if response.status_code != 200:
            return self._status_error(response, codes)
        try:
            result = json.loads(response.content)
        except ValueError:
            return _error("caliper.bad_response", "Caliper API answered with something that is not JSON.")
        return ToolOutput(ok=True, text=_truncate(json.dumps(result, sort_keys=True, ensure_ascii=False)), result=result)

    async def spec_text(self, preview: str = "") -> tuple[str, str] | None:
        """The open spec's canonical TOML and its path (relative to the workspace),
        or that of a preview, or None when no spec is open or the app cannot say.
        Used for diffs only."""
        out = await self.call("spec.get", {}, preview=preview)
        if not out.ok or not isinstance(out.result, dict):
            return None
        text = out.result.get("text")
        if not isinstance(text, str):
            return None
        path = out.result.get("path")
        spec_id = out.result.get("id")
        if not isinstance(path, str) or not path:
            path = f"applets/{spec_id}/{spec_id}.spec.toml" if isinstance(spec_id, str) and spec_id else "open.spec.toml"
        return text, path
