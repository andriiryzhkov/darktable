# darktable NOVA

NOVA is darktable's webview-based UI, providing a modern interface alongside
the traditional GTK interface. It runs as a separate binary (`darktable-nova`)
with two transport modes: direct (in-process) and server (IPC via socket).

## System Requirements

### Minimum

| Component | Requirement |
|-----------|------------|
| RAM       | 8 GB |
| CPU       | Quad-core (4 threads) |
| Storage   | SSD recommended |
| OS        | macOS 12+, Linux (with WebKitGTK), Windows 10+ |

### Recommended

| Component | Requirement |
|-----------|------------|
| RAM       | 16 GB |
| CPU       | 8+ threads |
| Storage   | NVMe SSD |
| GPU       | Any (WebGL-capable browser engine) |

### Difference from GTK darktable

NOVA has higher memory requirements than standard darktable because it runs
both the darktable processing engine and a webview browser engine:

- **Webview engine**: ~200-400 MB baseline (WebKit/Chromium)
- **Preview rendering**: Up to ~16 MB per preview frame (1920x1080 BGRA, double-buffered)
- **SHM buffers** (server mode): Additional ~16 MB for shared memory frame transfer
- **JavaScript UI**: ~50-100 MB for the React-based frontend

Standard darktable (GTK) can run on 4 GB systems. NOVA requires at least 8 GB
for comfortable editing.

### Low-memory tips

If you experience slowdowns or out-of-memory errors:

1. Close other memory-intensive applications
2. Use direct mode (default) instead of server mode to avoid SHM overhead
3. The preview resolution is automatically capped at 1920x1080 to limit memory use
4. Process fewer images in batch operations
5. Consider using standard darktable (GTK) if your system has less than 8 GB RAM

## Transport Modes

### Direct mode (default)

```
darktable-nova
```

Runs the darktable engine in-process. Lower latency, no socket overhead.
Best for local editing on the same machine.

### Server mode

```
darktable-nova --server
```

Spawns a separate `darktable-server` process and communicates via Unix socket.
Enables remote editing scenarios and process isolation.
