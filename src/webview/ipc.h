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

#include <glib.h>
#include <json-glib/json-glib.h>
#include <pthread.h>

// Connect to darktable-server Unix socket. Returns fd >= 0 on success, -1 on error.
int dt_ipc_connect(const char *socket_path);

// Send a JSON-RPC request and wait for the response.
// Thread-safe when protected by mutex.
// Returns the "result" field as a JSON string (caller must g_free), or NULL on error.
// On error, *out_error is set to a newly-allocated error string (caller must g_free).
char *dt_ipc_request(int fd, pthread_mutex_t *mutex,
                     const char *method, const char *params_json,
                     char **out_error);

// --- Event-aware IPC context (Phase 1A) ---

// Callback for server-pushed events.  Called from the reader thread.
typedef void (*dt_ipc_event_cb_t)(const char *event_name, const char *data_json, void *user_data);

typedef struct dt_ipc_context_t dt_ipc_context_t;

// Create an event-aware IPC context with a dedicated reader thread.
// The reader thread routes RPC responses to waiting callers and dispatches
// server-pushed events via event_cb.
dt_ipc_context_t *dt_ipc_context_new(int fd, dt_ipc_event_cb_t event_cb, void *event_user_data);

// Shut down the reader thread and free all resources.
void dt_ipc_context_free(dt_ipc_context_t *ctx);

// Send a JSON-RPC request and wait for the matching response (thread-safe).
// Same return semantics as dt_ipc_request().
char *dt_ipc_request2(dt_ipc_context_t *ctx,
                      const char *method, const char *params_json,
                      char **out_error);
