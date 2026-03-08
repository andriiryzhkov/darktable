# Webview Shell: Decision Analysis & Alternatives

> Should darktable's new UI use webview.h, or is there a better option?

**Date**: March 2026
**Context**: The webview UI shell (`darktable-nova`) uses [webview/webview](https://github.com/webview/webview) — a thin C/C++ library wrapping platform webview controls (WKWebView on macOS, WebView2 on Windows, WebKitGTK on Linux). This document challenges that decision.

---

## 1. Current Architecture

```
┌────────────────────────────────────────────────────┐
│  darktable-nova process                            │
│  ┌──────────────────────┐  ┌────────────────────┐  │
│  │  webview.h host      │  │  React SPA         │  │
│  │  (C, ~600 LOC)       │──│  (TypeScript)      │  │
│  │  • 50 JS↔C bindings  │  │  • Zustand stores  │  │
│  │  • 4-thread pool     │  │  • Bauhaus controls│  │
│  │  • IPC to server     │  │  • 89 IOP modules  │  │
│  │  • SHM frame client  │  │                    │  │
│  └──────────────────────┘  └────────────────────┘  │
│           │                                        │
│    Unix socket (JSON-RPC)                          │
│           │                                        │
│  ┌──────────────────────────────────────────────┐  │
│  │  darktable-server                            │  │
│  │  • 35 RPC routes                             │  │
│  │  • pixel pipeline                            │  │
│  │  • SHM double-buffered preview               │  │
│  │  • signal→event bridge                       │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

**Key property**: The server and the shell are cleanly separated by JSON-RPC over a Unix socket. The shell can be swapped without changing the server or the React SPA.

---

## 2. Alternatives Evaluated

### 2.1 CEF (Chromium Embedded Framework)

| Aspect | Details |
|--------|---------|
| **What** | Embeddable Chromium with a first-class C API (`cef_v8handler_t`, etc.) |
| **C compatibility** | Excellent — designed for embedding in C/C++ apps |
| **Bundle size** | ~100-150 MB (full Chromium) |
| **Platform support** | Linux, macOS, Windows, ARM64 |
| **Rendering** | Pixel-perfect consistency — same Chromium on all platforms |
| **License** | BSD — GPL-3 compatible |
| **Maintenance** | Active (Marshall Greenblatt, sponsored by Spotify) |
| **Used by** | Spotify, Steam, OBS Studio |

**Pros**: Eliminates all cross-platform rendering differences. The C API is production-proven. Off-screen rendering mode can composite into existing GTK windows.

**Cons**: +100 MB distribution size. Manual reference counting in C API is verbose. Chromium updates are frequent and large.

### 2.2 Tauri

| Aspect | Details |
|--------|---------|
| **What** | Rust application framework using system webviews |
| **C compatibility** | Poor — Rust framework, not embeddable in C |
| **Bundle size** | ~1 MB |
| **Rendering** | Same system-webview inconsistencies as webview.h |
| **License** | MIT — GPL-3 compatible |

**Verdict**: Not embeddable. Tauri is a standalone app framework. darktable would need to be restructured around Rust, gaining nothing over webview.h since both use the same underlying system webviews. ([GitHub issue #704](https://github.com/tauri-apps/tauri/issues/704) confirms embedding is unsupported.)

### 2.3 Electron

| Aspect | Details |
|--------|---------|
| **What** | Node.js + Chromium runtime |
| **C compatibility** | None — standalone runtime |
| **Bundle size** | ~150-200 MB |
| **Rendering** | Excellent consistency (bundled Chromium) |
| **License** | MIT — GPL-3 compatible |

**Verdict**: Heaviest option. +150 MB bundle, +200-500 MB RAM baseline. The existing JSON-RPC architecture would work (Electron connects to the Unix socket), but it's unjustifiable bloat for a photo editor that already needs all available RAM for pixel buffers.

### 2.4 Ultralight

| Aspect | Details |
|--------|---------|
| **What** | Lightweight HTML renderer (not a full browser) |
| **C compatibility** | Good — has C API |
| **Bundle size** | ~10-30 MB |
| **License** | **Proprietary core** — UltralightCore is closed-source |

**Verdict**: **GPL-3 incompatible.** Hard blocker regardless of technical merit.

### 2.5 Sciter

| Aspect | Details |
|--------|---------|
| **What** | Lightweight HTML/CSS engine with native scripting |
| **C compatibility** | Excellent — pure C API |
| **Bundle size** | ~5-10 MB |
| **License** | **Proprietary** — closed-source binary, commercial from $310 |

**Verdict**: **GPL-3 incompatible.** Also non-standard CSS/JS engine — the React SPA would need adaptation.

### 2.6 Qt WebEngine

| Aspect | Details |
|--------|---------|
| **What** | Chromium embedded in Qt framework |
| **C compatibility** | Poor — C++ only, requires Qt |
| **Bundle size** | ~100+ MB |
| **License** | LGPL v3 — GPL-3 compatible |

**Verdict**: darktable uses GTK. Adding Qt as a dependency for the webview alone is impractical. Same bundle size as CEF without the C API advantage.

### 2.7 Wails / Neutralino.js

Both are standalone app frameworks (Go / C++) that use system webviews internally. Neither is embeddable in an existing C application. Offer nothing over webview.h.

### 2.8 Saucer

| Aspect | Details |
|--------|---------|
| **What** | Modern C++23 webview library with multiple backend support |
| **C compatibility** | C++23 natively; C bindings exist ([saucer/bindings](https://github.com/saucer/bindings), 8 stars) |
| **Bundle size** | ~250 KB (uses system webviews) |
| **Platform support** | Linux (GTK4+WebKitGTK or Qt+QWebEngine), macOS (Cocoa+WKWebView or Qt), Windows (Win32+WebView2 or Qt) |
| **Rendering** | System webview (same inconsistencies as webview.h), but Qt/QWebEngine backend available as alternative |
| **License** | MIT — GPL-3 compatible |
| **Maintenance** | Single developer (1,571 of 1,574 commits). 805 stars. v8.0.0 released Dec 2025 |

**Pros**: Richer feature set than webview.h — thread-safe APIs, custom scheme handlers, 10+ typed events, resource embedding, Qt backend option on all platforms. The Qt/QWebEngine backend could provide Chromium consistency without bundling CEF.

**Cons**: Single-developer project (bus-factor risk). Rapid API churn (v6→v7→v8 in one year). C bindings are community-maintained with only 8 stars — may lag behind the C++ API. C++23 requirement means linking against C++ runtime. The richer API isn't needed for darktable's use case — the thin webview.h API is sufficient, and additional features (schemes, events) are handled by the JSON-RPC server layer.

**Verdict**: Interesting but risky. The Qt/QWebEngine fallback backend is a unique advantage, but the single-developer maintenance and API instability make it a poor choice for a long-lived project like darktable. webview.h is simpler, more stable, and sufficient.

### 2.9 Direct WebKitGTK / WKWebView / WebView2

Use platform APIs directly without a wrapper library.

**Verdict**: This is exactly what webview.h already does. Going "direct" means writing and maintaining three platform backends yourself — webview.h abstracts this in ~2K lines of well-tested code.

---

## 3. Hybrid Options (Keep Server, Swap Shell)

The JSON-RPC socket architecture makes shell swapping straightforward:

| Shell | Bundle | RAM | Rendering | Effort | Notes |
|-------|--------|-----|-----------|--------|-------|
| **webview.h (current)** | 0 MB | System | Varies by platform | Done | Production-ready |
| **CEF** | +100 MB | +50 MB | Identical everywhere | Medium | Rewrite ~600 LOC host |
| **Electron (as shell)** | +150 MB | +200 MB | Identical everywhere | Low | Trivial Node.js app |
| **System browser** | 0 MB | 0 MB | User's browser | None | No native window control |
| **PWA / --app mode** | 0 MB | Shared | Chrome | None | Requires Chrome installed |

---

## 4. Comparative Summary

| Criterion | webview.h | CEF | Saucer | Electron | Others |
|-----------|-----------|-----|--------|----------|--------|
| **C-embeddable** | Yes | Yes | Via C bindings | No | Varies |
| **GPL-3 compatible** | Yes (MIT) | Yes (BSD) | Yes (MIT) | Yes (MIT) | Ultralight/Sciter: No |
| **Bundle overhead** | 0 MB | ~100 MB | ~250 KB | ~150 MB | — |
| **Cross-platform** | Yes (3 backends) | Yes (Chromium) | Yes (3+Qt) | Yes (Chromium) | — |
| **Rendering consistency** | Low | High | Low (Qt option) | High | — |
| **IPC model** | webview_bind → C callback | CefV8Handler | Reflection + schemes | Node IPC | — |
| **Maturity** | Good | Excellent | Early (single dev) | Excellent | — |
| **DevTools support** | Platform-dependent | Always available | Platform-dependent | Always available | — |

---

## 5. webview.h: Known Pain Points

These are real issues with the current approach, not theoretical:

1. **Rendering inconsistency**: WebKitGTK, WKWebView, and WebView2 render CSS differently. Flexbox gaps, scrollbar styling, font rendering, and backdrop-filter all vary. Testing must cover all three engines.

2. **DevTools**: Available on macOS (Safari) and Linux (WebKitGTK inspector) but awkward on Windows (WebView2 requires specific flags). No unified debugging story.

3. **Web API gaps**: WebKitGTK lags behind Chrome on newer CSS/JS features. Features like `@container` queries, `color-mix()`, or `View Transitions API` may not be available on all platforms simultaneously.

4. **Security surface**: `file://` origin gives the webview broad filesystem access. A malicious SVG thumbnail or XSS in a module parameter could read arbitrary files. Mitigation exists (`path_validation.h`) but is defense-in-depth, not a sandbox.

5. **No off-screen rendering**: webview.h creates its own window. Cannot composite the web content into an existing GTK widget tree (relevant for a potential future where the webview UI coexists with GTK panels).

6. **Binding limitations**: `webview_bind` only supports string (JSON) arguments. No binary data transfer — preview frames require a separate HTTP frame server.

---

## 6. webview.h: Strengths

1. **Zero bundle overhead**: Uses the browser engine already installed on every OS. No downloads, no version management.

2. **Minimal integration surface**: ~600 lines of C host code. The webview.h API is 10 functions. Low maintenance burden.

3. **Correct abstraction level**: webview.h solves exactly one problem — "give me a window with a web renderer and a way to call between JS and C." It does not impose an application framework.

4. **Clean architecture enabler**: The simplicity of webview.h forced a clean JSON-RPC server separation, which enables remote editing, headless testing, and potential direct-transport mode.

5. **C API**: First-class C API matches darktable's codebase. No FFI bridges, no language runtime dependencies.

6. **Active maintenance**: The webview/webview project is actively maintained with regular releases tracking platform webview updates.

---

## 7. Decision Matrix

| Weight | Criterion | webview.h | CEF |
|--------|-----------|-----------|-----|
| High | GPL-3 compatibility | Pass | Pass |
| High | C API availability | Pass | Pass |
| High | Bundle size (photo editors need RAM, not disk) | 0 MB | +100 MB |
| Medium | Cross-platform rendering consistency | Varies | Identical |
| Medium | Implementation effort (already done vs rewrite) | Done | ~2 weeks |
| Medium | DevTools / debugging experience | Platform-dependent | Always Chrome DevTools |
| Low | Off-screen rendering capability | No | Yes |
| Low | Binary data in bindings | No (workaround exists) | Yes |

---

## 8. Recommendation

**Keep webview.h.** It is the only option that simultaneously satisfies:
- Embeddable in a C codebase (not a framework that takes over `main()`)
- GPL-3 compatible
- Zero bundle size overhead
- Cross-platform (Linux, macOS, Windows)

**CEF is the only credible alternative**, trading +100 MB bundle for rendering consistency. The existing architecture makes a future swap straightforward if cross-platform rendering differences become a real (not hypothetical) problem.

**The architecture's true strength is the server separation**, not the webview choice. The JSON-RPC socket protocol means the shell is a thin, replaceable layer — a deliberate design decision that de-risks the webview.h choice.

### When to reconsider

- If WebKitGTK falls significantly behind on CSS features needed by the UI
- If cross-platform visual testing reveals unacceptable rendering divergence
- If a future feature requires off-screen rendering or GTK embedding
- If webview.h maintenance stalls

None of these are currently blocking.
