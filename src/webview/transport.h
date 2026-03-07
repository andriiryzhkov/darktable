/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

/*
 * Transport vtable: abstracts communication between the webview UI and
 * libdarktable/server.
 *
 * Two planned implementations:
 *
 *   1. IPC transport (current architecture)
 *      - Webview UI runs in a separate process from darktable-server
 *      - Communication via Unix domain socket using JSON-RPC
 *      - Preview frames delivered via shared memory (SHM)
 *      - Events pushed from server → client via the socket
 *
 *   2. Direct transport (future, Alternative E)
 *      - Webview UI runs in-process with libdarktable
 *      - Function calls go directly to libdarktable APIs (no serialization)
 *      - Preview frames read directly from pipeline backbuf (no SHM)
 *      - Events delivered via callback (no socket)
 *
 * The vtable uses a generic JSON-based interface for most operations
 * (matching the existing JSON-RPC protocol), plus specialized hooks for
 * performance-critical paths (preview frames, events) where the direct
 * transport can avoid serialization entirely.
 *
 * All methods are synchronous from the caller's perspective — the binding
 * layer handles async dispatch via the thread pool. The transport itself
 * blocks until the operation completes.
 */

#include <glib.h>
#include <stdint.h>

/* Forward declarations */
typedef struct dt_webview_transport_t dt_webview_transport_t;

/* Event callback: called by the transport when the server pushes an event.
 * event_name and json_data are owned by the caller and valid only during
 * the callback invocation. */
typedef void (*dt_transport_event_cb)(const char *event_name,
                                     const char *json_data,
                                     void *user_data);

/* Preview frame data returned by get_preview_frame.
 * For IPC transport: pixels are copied from SHM.
 * For direct transport: pixels point directly into the pipeline backbuf
 * (caller must NOT free; valid until next pipeline run). */
typedef struct dt_transport_frame_t
{
  const uint8_t *pixels;   /* BGRA8 pixel data */
  int width;
  int height;
  uint64_t sequence;       /* frame sequence number */
  gboolean owned;          /* TRUE if pixels must be g_free'd by caller */
} dt_transport_frame_t;


/* ── Transport vtable ────────────────────────────────────────── */

struct dt_webview_transport_t
{
  /* Opaque implementation data (IPC context, direct lib handle, etc.) */
  void *data;

  /* ── Generic JSON-RPC call ─────────────────────────────────
   *
   * The primary interface for most operations. Maps 1:1 to the
   * existing server protocol methods.
   *
   * Parameters:
   *   method      - JSON-RPC method name (e.g. "catalog.query", "develop.set_params")
   *   params_json - JSON object string with method parameters, or NULL for no params
   *   error       - on failure, set to a g_malloc'd error message (caller frees)
   *
   * Returns:
   *   On success: g_malloc'd JSON string with the result (caller frees)
   *   On failure: NULL, with *error set
   *
   * This covers all operations in these categories:
   *
   *   System:     system.ping
   *   Catalog:    catalog.query, catalog.get_thumbnail, catalog.get_thumbnails,
   *               catalog.get_tags, catalog.get_filmrolls, catalog.import,
   *               catalog.copy_import, catalog.get_file_thumbnail,
   *               catalog.get_collection_values, catalog.check_imported
   *   Develop:    develop.open, develop.close, develop.get_modules,
   *               develop.get_history, develop.get_params, develop.set_params,
   *               develop.commit_params, develop.reset_params,
   *               develop.request_preview, develop.cancel_pipeline,
   *               develop.sample_pixels, develop.select_history,
   *               develop.compress_history, develop.truncate_history,
   *               develop.delete_history, develop.list_presets,
   *               develop.apply_preset, develop.store_preset,
   *               develop.delete_preset, develop.new_instance,
   *               develop.delete_instance, develop.move_instance,
   *               develop.rename_instance, develop.get_introspection
   *   Config:     config.get, config.set
   *   Export:     export.image
   */
  char *(*call)(dt_webview_transport_t *self,
                const char *method,
                const char *params_json,
                char **error);

  /* ── Preview frame access ──────────────────────────────────
   *
   * Performance-critical path — avoids JSON serialization.
   *
   * IPC transport:  reads from shared memory, returns owned copy
   * Direct transport: returns pointer into pipeline backbuf (zero-copy)
   *
   * Returns FALSE if no frame is available.
   */
  gboolean (*get_preview_frame)(dt_webview_transport_t *self,
                                const char *session_id,
                                dt_transport_frame_t *out_frame);

  /* ── Event subscription ────────────────────────────────────
   *
   * Register a callback for server-pushed events.
   *
   * IPC transport:  reader thread receives events from socket, calls cb
   * Direct transport: signal handlers call cb directly
   *
   * Only one callback is supported. Calling again replaces the previous one.
   * Pass NULL callback to unsubscribe.
   */
  void (*set_event_callback)(dt_webview_transport_t *self,
                             dt_transport_event_cb callback,
                             void *user_data);

  /* ── Lifecycle ─────────────────────────────────────────────
   *
   * destroy: release all resources, close connections, free self.
   * After destroy(), the pointer is invalid.
   */
  void (*destroy)(dt_webview_transport_t *self);
};


/* ── Convenience macros ──────────────────────────────────────── */

#define dt_transport_call(t, method, params, err) \
  ((t)->call((t), (method), (params), (err)))

#define dt_transport_get_preview_frame(t, sid, frame) \
  ((t)->get_preview_frame((t), (sid), (frame)))

#define dt_transport_set_event_callback(t, cb, ud) \
  ((t)->set_event_callback((t), (cb), (ud)))

#define dt_transport_destroy(t) \
  ((t)->destroy((t)))


/* ── Factory functions (implemented in transport_ipc.c / transport_direct.c) ── */

/* Create an IPC transport connected to a darktable-server via Unix socket.
 * socket_fd: connected socket file descriptor (transport takes ownership)
 * Returns NULL on failure. */
dt_webview_transport_t *dt_transport_ipc_new(int socket_fd);

/* Create a direct transport using in-process libdarktable.
 * Requires darktable to be initialized (dt_init() called).
 * Returns NULL on failure. */
dt_webview_transport_t *dt_transport_direct_new(void);
