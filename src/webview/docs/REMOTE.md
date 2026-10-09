# NOVA Remote Control Plan

> Plan for running Nova against a darktable on another machine: the
> server on a desktop, the client on a thin laptop or an Android tablet
> over the local network. Nothing here is implemented yet.

**Date**: October 2026

Related: darktable-org/darktable#21340 (RFC: control of darktable over
network)

---

## 1. Target

- **Desktop**: runs `darktable-server`, listening on the local network.
- **Tablet or thin laptop**: opens `http://desktop:port` in a browser. The
  server serves `ui/dist` and speaks JSON-RPC 2.0 over a WebSocket.
- A native Android client should not be needed: the Nova UI is already a
  web app. A thin laptop can use the browser too, or
  `darktable-nova --connect host` later.

## 2. What already fits

- Every host call in the UI goes through `ui/src/api/commands.ts` (67
  `window.*` bindings). That is a single seam for a second transport.
- Previews are already fetched over HTTP (`ui/src/stores/developStore.ts`),
  so they only need a configurable base URL.
- `darktable-server` is already a separate process with token auth and
  pipeline cancellation (`develop.cancel_pipeline`).

## 3. What has to change

1. **Protocol**: JSON-RPC 2.0 over WebSocket. Browsers cannot open raw TCP
   or Unix sockets. One WebSocket message is one JSON-RPC message, so the
   4-byte length prefix stays only on the Unix socket transport. Events
   become JSON-RPC notifications.
2. **Images**: shared memory works on one machine only. Previews are
   encoded by the server (JPEG or WebP; roughly 200-400 KB at 1920x1080).
   Thumbnails move out of base64-in-JSON (the reason for the 16 MB message
   cap in `server_protocol.h`) to `GET /thumb/{imgid}/{size}` with ETags,
   which also helps local mode.
3. **Same-filesystem assumptions**: import paths, the native folder picker
   (runs on the client), `copy_import` and export destinations. Remote use
   needs server-side directory browsing and export delivered as a
   download.
4. **Security**: on a LAN anyone on the network can reach the port.
   - bind to loopback by default; require an explicit `--listen 0.0.0.0`
   - pair devices with a code or QR shown on the desktop instead of a
     fixed token
   - TLS, since a token over plain `ws://` can be sniffed
   - the hand-written HTTP frame server in `src/webview/bindings.c` is
     fine on loopback but should not face a network
5. **Latency**: coalesce slider updates (send only the latest value),
   cancel the in-flight render, and send a smaller preview while dragging.
6. **Several clients**: desktop and tablet on the same image need shared
   edit sessions and events fanned out to every client.

## 4. Phases

### Phase 1: browser on the same desktop (loopback only)

Each step builds and leaves the native Nova window working.

**Step 1: JSON-RPC 2.0 envelope (small)**

- `src/server/server_protocol.c`: add `"jsonrpc": "2.0"`; send exactly one
  of `result` or `error`; accept numeric ids; no reply to requests without
  an id; events as notifications (`method` + `params` instead of
  `event` + `data`)
- `src/webview/ipc.c` and `src/webview/direct_transport.c`: the two places
  that parse responses and events
- tests: `src/tests/unittests/webview/test_protocol.c`, `test_transport.c`,
  `src/tests/unittests/webview_server/`, `src/server/test_client.py`
- the UI is untouched: the host returns only `result` to JS
  (`bindings.c`, `webview_return`) and delivers events through
  `webview_eval`

**Step 2: transport seam in the UI (medium)**

Each of the 67 bindings has its own handler in `src/webview/bindings.c`
(about 3,400 lines), largely converting positional JS arguments into
named server params. A browser cannot use them. Preferred approach:

- `commands.ts` calls a generic `call(method, params)` everywhere
- native window: one generic `rpc` binding forwards to the transport
- browser: the same call goes over a WebSocket
- most per-binding adapters in `bindings.c` are deleted; host-only
  bindings stay native: `pickFolder`, `windowStartDrag`, `windowZoom`,
  `getPlatformInfo`, the frame port

Rejected alternative: re-implementing the 67 adapters in TypeScript, which
keeps two copies of every argument mapping.

**Step 3: HTTP and WebSocket in darktable-server (medium-large)**

- serve `ui/dist`, JSON-RPC over WebSocket, event delivery to the browser
- previews as JPEG over HTTP instead of shared memory; thumbnails over
  HTTP
- move the filesystem bindings (`listFolders`, `listFiles`, `getHomePath`,
  `getFileThumbnail`) from the host into server methods; they are already
  C (`g_dir_open`), so they move rather than get rewritten

Not needed in phase 1: on the same machine server paths are valid, so
import and export work unchanged. Only the native folder picker is
unavailable in the browser; the in-app folder browser replaces it once it
is a server method.

### Phase 2: local network

- server-side directory browsing for import; export as a download
- `--listen` on a LAN address, device pairing, TLS
- latency work from section 3.5

### Phase 3: several clients

- shared edit sessions per image, events fanned out to all clients
- mmcc-xx's darktable-api prototype already does this
  (https://github.com/mmcc-xx/darktable/tree/darktable-api); worth
  converging rather than building it twice

## 5. Open questions

- **HTTP/WebSocket library**: libsoup 3 provides an HTTP server,
  WebSocket and TLS, is GLib-based, and WebKitGTK already depends on it on
  Linux. It is still a new dependency of the darktable build and needs
  maintainer agreement. The alternative is extending the hand-written HTTP
  code, which is not advisable once it faces a network.
- **Bindings audit**: the step 2 estimate rests on the binding names and a
  sample handler (`on_catalog_query`). Before committing to it, sort the 67
  bindings into plain forwarding, host-only, and handlers with real logic
  (for example the import worker thread, `_import_images_worker`). Logic
  in the third group has to move into the server, and that is what would
  make step 2 larger.
- **One JSON-RPC implementation**: darktable-mcp already speaks JSON-RPC
  2.0 (`src/mcp/mcp_jsonrpc.c`). Sharing it with `darktable-server` would
  avoid two dialects in the tree, but touches `src/mcp`.
