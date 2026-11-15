# Caliper API

Caliper API is an HTTP interface to the running app. Version 0 lists the applets, launches one, takes data streams over [wire protocol v3](wire-protocol.md), and returns a capture of the frame the app draws. It listens on 127.0.0.1 only, behind a session token, and it is off unless the app starts with `--api`. The server is in `src/host/api/api_server.h`, and `src/host/api/app_api.h` starts it for the app and for `caliper shot`.

!!! note "Status"
    Version 0, built and tested on macOS. A capture works in the app's window on Metal and in the windowless mode, `caliper shot --api`; on the GL and Vulkan renderers the app in its window answers a capture with 501. Driving the app's state from outside, beyond launching an applet, comes with the command layer of [Phase 3](../plans/roadmap.md#phase-3-agent-built-panels).

| Method and path | What it does |
|---|---|
| `GET /v0/info` | The host's version, ABI epoch, services and SDK |
| `GET /v0/applets` | The applets the app lists, as `caliper list` shows them |
| `POST /v0/applets/{id}/launch` | Opens that applet's page |
| `GET /v0/capture.png` | The frame the app draws, as a PNG |
| `GET /v3/stream` | The WebSocket of wire protocol v3, with the app as the hub, at the protocol's own path |
| `GET /v0/stream` | The same WebSocket, under the API's prefix |

## Starting it

`--api` turns the API on for the app, for `caliper dev` and for `caliper shot`, and so does the environment variable `CALIPER_API=1`. `--api-port <n>` chooses the port, from 0 to 65535, and needs `--api`; without it the API listens on port 7341, and 0 takes any free port. [Caliper CLI](cli.md#caliper-api) lists the options.

```sh
caliper --api
caliper dev build --api --api-port 0
caliper shot --applets build --launch com.example.my-viewer --frames 216000 --out page.png --api
```

When it listens, the process prints one line with the address and the session token, and writes the same in the **API file**:

```text
[api] Caliper API on http://127.0.0.1:7341/v0 with the session token <token> (also in <API file>)
```

The API file is `api.json` in the app-data folder ([Folders and environment variables](environment.md)). The process writes it with mode 0600, so that only its owner can read it, and removes it when it exits, as `caliper serve` does with its hub file. It holds one JSON object:

| Key | Value |
|---|---|
| `version` | 1 |
| `pid` | The process id |
| `port` | The port the API listens on |
| `token` | The session token |
| `url` | `http://127.0.0.1:<port>/v0` |
| `stream_url` | `ws://127.0.0.1:<port>/v3/stream`, without the token |

The session token is new for each run: 32 characters of `A-Z`, `a-z`, `0-9`, `_` and `-` from 24 random bytes. When the port is taken, the app starts without the API and says so on standard error and in the Activity pane, and `caliper shot` exits with 1.

## The windowless mode

`caliper shot` draws the app's applet page into an offscreen canvas and shows nothing on screen ([caliper shot](cli.md#caliper-shot)). With `--api`, it serves the API while it draws its frames, so it is the app without a window: `--frames` sets how long it runs, at about 60 frames a second, and SIGINT (Ctrl-C) or SIGTERM ends the frames early. Either way it then writes its image of the applet it shows, as usual, stops the API, removes the API file and exits with 0. The API file goes to the app-data folder also when `shot` keeps its stores in a temporary folder.

A capture is read from the offscreen canvas at the end of a frame, with the read-back that `caliper shot` uses for its image.

## Security

Every request gets the checks of the stream ([The upgrade request](wire-protocol.md#the-upgrade-request)):

- The server listens on 127.0.0.1 only.
- The `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`, with the API's port. Anything else is refused with 403, as a guard against DNS rebinding.
- An `Origin` header, when the request has one, must be one that the protocol accepts: `null`, an `http` or `https` origin on `127.0.0.1`, `localhost` or `[::1]`, or `vscode-webview://`. Any other origin is refused with 403.
- The session token must come with the request. A request carries it in an `Authorization: Bearer <token>` header. A GET may carry it as the query parameter `token` instead, which is how the WebSocket and a link opened in a browser carry it. A POST takes the header only, so a state-changing request never has the token in its address. The token is compared in constant time. A missing or wrong token is refused with 401 and `WWW-Authenticate: Bearer`, and on the stream with 403, as the protocol says.

The API serves nothing beyond its endpoints: no files, no command layer and no way to run code. A request with a body is refused with 413, and launch takes only an applet id in its path that the app's loader lists. Every response carries `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`, and the connection closes after it.

## Endpoints

### GET /v0/info

What `caliper info` prints, as JSON. `sdk`, `libtorch` and `ui_stack` are `null` when they are unknown or absent.

```json
{"abi_epoch":2,"host":"0.6.0","libtorch":"2.5.1","sdk":"0.2.0","services":["caliper.artifacts.v1","caliper.data.v1","caliper.log.v1","caliper.ui.v1"],"ui_stack":"03c557182adc5cd9"}
```

The list of services is shortened here.

### GET /v0/applets

What `caliper list` prints for the applets the running app found, in the same order: the search order, then the id, then the highest version first. `status` is the word that `caliper list` prints, such as `READY`, `ACTIVE` for the applet whose page is open, `REFUSED` or `SHADOWED`, and `reason` is the text that `caliper list` prints under it, empty for a `READY` applet.

```json
{"applets":[{"folder":"/path/to/build/applets","id":"com.example.my-viewer","reason":"","status":"ACTIVE","version":"0.1.0"}]}
```

### POST /v0/applets/{id}/launch

Opens the page of the applet with that manifest id, as a click on its row on the landing screen does. The page shown before goes first, as Home takes it: an open applet is torn down, and an attached live session is detached. When the new applet fails to launch, the app stays on the landing screen with the reason in the Activity pane, and the windowless mode launches the applet it showed before again.

```json
{"already_open":false,"id":"com.example.my-viewer","status":"ACTIVE","version":"0.1.0"}
```

| Status | When |
|---|---|
| 200 | The applet's page is open. `already_open` is true when it was open before the request. |
| 400 | The id is not 1 to 128 characters of `A-Z`, `a-z`, `0-9`, `.`, `_` and `-`. |
| 404 | No folder in the search order holds an applet with that id. |
| 409 | The applet cannot be launched, with the status and reason that `caliper list` shows, or it failed to launch, with the loader's reason. |

### GET /v0/capture.png

The frame the app draws, as an opaque PNG, read back at the end of the next frame without drawing an extra one:

- In the app's window, the PNG is the frame the window presents, at the window's framebuffer size in pixels (twice the window's size in points on a Retina display). This works with the window hidden too ([`--hidden`](cli.md#the-hidden-window)), which is how tests and agents drive the whole app with nothing on screen. The renderer copies that frame on the GPU as it presents it, and a later frame picks the copy up, so the frame loop never waits for it. This works on Metal. On the GL and Vulkan renderers the window's frame cannot be read back yet, and a capture is answered with 501 and the reason.
- In the windowless mode, the PNG is the offscreen canvas at the end of a frame, of the size `caliper shot --size` gives, 1600 by 1000 by default.

A capture waits for the app's frames like every call that needs the frame thread, so it is answered with 504 when no frame is drawn within 5 seconds. The PNG is encoded on a thread of its own, so the frame loop does not wait for the encoder either.

### GET /v3/stream

The WebSocket of [wire protocol v3](wire-protocol.md), with the app in the hub role. It is the hub of [`caliper serve`](../guides/run-a-hub.md) on the API's port: producers connect to it and each becomes a session, and viewers list the sessions and attach to one. Everything the protocol states for a hub applies, with its bounds and its flow control. It is served at the protocol's own path, so every client of the protocol attaches to it unchanged: the app's NetSource, `caliper-viewer --url`, the link field of the app's attach pane and the `<caliper-view>` element. The upgrade is checked for its path, `Host`, `Origin` and the `token` parameter in the protocol's order, so a client gives the token in the address:

```text
ws://127.0.0.1:7341/v3/stream?token=<token>
```

`GET /v0/stream` is an alias of it, with the same checks, for clients that keep every address under the API's prefix. The stream carries data and viewer events only, never host commands.

## Threads and limits

One thread runs the hub's loop and answers every request; it never waits for the app. A request that reads or changes the app's state, which is the list of applets, a launch or a capture, waits for the app's frame thread, which takes it at the start of a frame, between two frames. A launch runs there, as `CALIPER_AUTOLAUNCH` runs before the first frame.

| Limit | Value | Answer beyond it |
|---|---|---|
| Time a request waits for the frame thread | 5 s | 504; a launch that is slower than that may still complete |
| Requests waiting for the frame thread at once | 8 | 503 with `Retry-After: 1` |
| Size of a request head | 16 KiB | 431 |
| TCP connections, the stream included | 512 | The connection is closed |

The stream's own bounds are those of the hub; see [Run a hub](../guides/run-a-hub.md). When the app exits, requests still waiting get 503.

## Errors

An error is answered with a JSON object `{"error": "<what went wrong>"}`.

| Status | When |
|---|---|
| 400 | A launch names something that cannot be an applet id. |
| 401 | The session token is missing or wrong. |
| 403 | The `Host` or the `Origin` header is refused. |
| 404 | No such endpoint, or no applet with that id. |
| 405 | The endpoint takes another method; `Allow` names it. |
| 409 | The applet cannot be launched, or failed to launch. |
| 413 | The request has a body. |
| 426 | A GET of `/v3/stream` or `/v0/stream` that is not a WebSocket upgrade. |
| 501 | A capture in the app's window on the GL or Vulkan renderer, which cannot read the window's frame back. |
| 503 | Too many requests wait for the app, or the app is closing. |
| 504 | The app did not take the request within 5 s. |

## Examples

With the token in `TOKEN`, from the line the app printed or from the API file:

```sh
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:7341/v0/info
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:7341/v0/applets
curl -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:7341/v0/applets/com.example.my-viewer/launch
curl -o frame.png -H "Authorization: Bearer $TOKEN" http://127.0.0.1:7341/v0/capture.png
```

In Python, the token can be read from the API file:

```python
import json, pathlib, urllib.request

api = json.loads((pathlib.Path.home() / "Library/Application Support/Caliper/api.json").read_text())
req = urllib.request.Request(api["url"] + "/applets", headers={"Authorization": "Bearer " + api["token"]})
print(json.load(urllib.request.urlopen(req)))
```
