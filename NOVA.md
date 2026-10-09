# darktable Nova

Nova is an experimental alternative user interface for darktable. The UI is a
React + TypeScript application running in a native webview (WKWebView on
macOS, WebKitGTK on Linux), and it drives the same engine as the GTK
application: `libdarktable`, the pixelpipe, the processing modules and the
library database.

Nova adds new binaries and does not change the GTK interface. Building it is
opt-in and disabled by default.

> **Status: prototype.** It is good enough to browse a library and edit
> images, but it is not feature-complete and not ready for end users. Use a
> separate config directory (see [Run](#run)) and keep backups of anything
> you care about.

## What is there

- **lighttable**: thumbnail grid, import, collections and collection
  filters, selection, ratings and color labels, image information, tagging,
  metadata, geotagging, styles, export
- **darkroom**: preview rendering, histogram and scopes, navigation and zoom,
  history with undo/redo, snapshots, presets, multiple instances
- **processing modules**: exposure, white balance, color calibration,
  sigmoid, demosaic, raw black/white point, orientation and input/output color
  profile have hand-built UIs. All other modules get a generic UI generated
  from their parameter introspection.
- **masks**: circle, ellipse, gradient, path and brush drawn masks, mask
  manager, opacity and refinement controls in the blending section

## How it works

Nova has three layers:

```
 ui/  React UI ──── JS bindings (window.catalogQuery(), window.developSetParams(), …)
                 ◀── events + preview frames (loopback HTTP)
        │
 src/webview/  darktable-nova (native window, bindings, frame server)
        │
        ├── direct mode (default): handlers run in-process
        └── server mode (--server): Unix socket ──▶ darktable-server
        │
 src/server/  request handlers (catalog.*, develop.*, export.*, config.*)
        │
 libdarktable  (headless: dt_init() without GUI)
```

- **UI (`ui/`)**: React 18, TypeScript, Vite, Tailwind CSS 4 and zustand
  stores. It calls functions that the host injects into `window` and
  receives server events through a typed event bus.
- **Host (`src/webview/`)**: the `darktable-nova` executable. It opens the
  native window through the [webview](https://github.com/webview/webview)
  library, registers the JS bindings and runs a small loopback HTTP server
  that delivers preview frames to the UI. It also provides the platform
  titlebar, the splash screen and the native folder picker
  ([nativefiledialog-extended](https://github.com/btzy/nativefiledialog-extended)).
- **Server (`src/server/`)**: JSON request handlers on top of
  `libdarktable`. The same code is compiled into `darktable-nova` for direct
  mode and into the standalone `darktable-server` for server mode.

There are two transport modes:

| Mode | How | Why |
|------|-----|-----|
| direct (default) | the handlers run inside the `darktable-nova` process | lowest latency, single process |
| server (`--server`) | `darktable-nova` starts `darktable-server`, connects over a Unix socket and authenticates with a random token; preview frames travel through shared memory | process isolation, groundwork for remote clients |

## Repository layout

| Path | Contents |
|------|----------|
| `ui/` | frontend (React/TypeScript), `npm` project |
| `src/webview/` | `darktable-nova` host: `main.c`, `bindings.c`, transports, platform titlebars, splash |
| `src/server/` | request handlers and the `darktable-server` executable |
| `src/webview/docs/` | design documents (architecture, testing, performance, masking, reviews) |
| `src/tests/unittests/webview/` | C unit tests (protocol, transport, path validation) |
| `src/tests/unittests/webview_server/` | pytest tests against a running `darktable-server` |
| `src/tests/perf/` | transport benchmarks |
| `tools/webview-mcp/` | MCP server for automating the Nova UI from Claude Code |
| `src/external/webview`, `src/external/nativefiledialog-extended` | git submodules |

## Requirements

- Everything needed to [build darktable](README.md#dependencies) itself
- macOS, or Linux with WebKitGTK 4.1 and GTK 3 development packages
  (`libwebkit2gtk-4.1-dev` on Debian/Ubuntu, `webkit2gtk4.1-devel` on Fedora).
  Windows is not supported yet: the host relies on `fork()`, `mmap()` and
  Unix sockets.
- Node.js 20 or newer, and npm
- 8 GB RAM minimum, 16 GB recommended
  ([details](src/webview/docs/NOVA.md#system-requirements))

## Build

Fetch the two additional submodules:

```bash
git submodule update --init --recursive src/external/webview src/external/nativefiledialog-extended
```

Configure and build darktable with Nova enabled:

```bash
./build.sh --enable-nova
```

or, driving cmake by hand:

```bash
cmake -B build -DUSE_NOVA=ON
cmake --build build -j
```

This produces `build/bin/darktable-nova` and `build/bin/darktable-server`.

| CMake option | Default | Meaning |
|--------------|---------|---------|
| `USE_NOVA` | `OFF` | build `darktable-nova` and `darktable-server` (the backend for `--server` mode) |
| `BUILD_WEBVIEW_DIRECT` | `ON` | link `libdarktable` into `darktable-nova` for direct mode |

Install the frontend dependencies:

```bash
cd ui
npm ci
```

## Run

Run `darktable-nova` from the **repository root**: by default it looks for the
frontend at `ui/dist` relative to the current directory.

Give Nova its own config directory. darktable locks the library while it is
open, so Nova and GTK darktable cannot share one at the same time, and an
experimental UI should not touch your main library anyway.

### Development mode (recommended)

Start the Vite dev server, which supports hot reload:

```bash
cd ui
npm run dev          # serves http://localhost:5173
```

In a second terminal, from the repository root:

```bash
./build/bin/darktable-nova --dev --core --configdir ~/.config/darktable-nova
```

On macOS with Homebrew, if GLib complains about missing settings schemas,
prefix the command with
`GSETTINGS_SCHEMA_DIR=/opt/homebrew/share/glib-2.0/schemas`.

### Server mode

```bash
./build/bin/darktable-nova --dev --server --core --configdir ~/.config/darktable-nova
```

`darktable-nova` looks for `darktable-server` next to its own binary. Use
`--server-bin PATH` or `DT_SERVER_BIN` to point elsewhere.

### Production build

```bash
cd ui && npm run build                       # writes ui/dist
cd .. && ./build/bin/darktable-nova --core --configdir ~/.config/darktable-nova
```

This does not work at the moment; see [Known issues](#known-issues).

### Options

`darktable-nova [OPTIONS] [--core DARKTABLE_OPTIONS]`

| Option | Meaning |
|--------|---------|
| `--dev` | load the UI from the Vite dev server at `http://localhost:5173` |
| `--frontend-dir DIR` | load the production build from `DIR` (default `ui/dist`, or `DT_FRONTEND_DIR`) |
| `--server` | use server mode instead of direct mode |
| `--server-bin PATH` | path to `darktable-server` (implies `--server`) |
| `--core …` | everything after it goes to the darktable core, e.g. `--configdir`, `--library`, `--cachedir`, `--disable-opencl`, `-d <domain>` |

### Running darktable-server on its own

```bash
./build/bin/darktable-server --socket /tmp/dt.sock --core --configdir ~/.config/darktable-nova
```

The server prints `SOCKET=` and `TOKEN=` lines on stdout. A client must send
`{"method": "auth", "params": {"token": "…"}}` as its first message. Set
`DARKTABLE_SERVER_TOKEN` to use a fixed token. `src/server/test_client.py`
is a minimal Python client.

## Tests

| What | Command |
|------|---------|
| C unit tests (needs cmocka) | `cmake -B build -DBUILD_TESTING=ON && cmake --build build && ctest --test-dir build -R 'test_(protocol\|transport\|security)'` |
| UI unit tests | `cd ui && npm test` |
| server integration tests | `python3 -m pytest src/tests/unittests/webview_server` (uses `build/bin/darktable-server`) |
| transport benchmarks | see [PERFORMANCE.md](src/webview/docs/PERFORMANCE.md) |

## UI automation

`tools/webview-mcp/` is an MCP server that lets Claude Code drive a running
Nova window: evaluate JavaScript, query the DOM, click, type and take
screenshots. It is registered in `.mcp.json` at the repository root. To set it
up, run:

```bash
cd tools/webview-mcp && npm ci
```

It talks to the `/test/eval` endpoint of the frame server. The port is found
through `$TMPDIR/darktable_test_port`, which `darktable-nova` writes at
startup.

## Known issues

As of October 2026:

- **Production mode shows a blank window.** The UI is loaded from `file://`,
  the Vite build uses absolute `/assets/…` paths, and WebKit refuses to load
  ES module scripts from `file://` origins anyway. Use `--dev` until the host
  serves `ui/dist` over HTTP.
- **`npm run build` fails at the `tsc -b` step** with type errors, most of
  them in test fixtures. `npx vite build` still produces a bundle.
- **Killing `darktable-nova` leaves `darktable-server` running** in server
  mode, and that process keeps the library locked. Close Nova from the window,
  or kill the server by hand.
- **`/test/eval` is always enabled.** It executes arbitrary JavaScript in the
  Nova window for anyone who can reach the loopback port. This is acceptable
  for development, but not for a release.
- The branch is based on darktable master from 2026-03-01.

## Further reading

- [ARCHITECTURE.md](src/webview/docs/ARCHITECTURE.md): full architecture,
  wire protocol, session model, data flows
- [ARCHITECTURE_2.md](src/webview/docs/ARCHITECTURE_2.md): transport
  abstraction, frame delivery, event system
- [TESTING.md](src/webview/docs/TESTING.md): testing strategy
- [PERFORMANCE.md](src/webview/docs/PERFORMANCE.md): benchmarks
- [MASKING.md](src/webview/docs/MASKING.md): how masks and distortions map
  between darktable and Nova
- [WEBVIEW_ALTERNATIVES.md](src/webview/docs/WEBVIEW_ALTERNATIVES.md): why
  webview/webview was chosen
- [NOVA.md](src/webview/docs/NOVA.md): system requirements and memory use
- [REMOTE.md](src/webview/docs/REMOTE.md): plan for remote control over
  the local network
