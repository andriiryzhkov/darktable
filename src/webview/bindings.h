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

#include <webview/api.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/types.h>

#include "ipc.h"

#define DT_WEBVIEW_MAX_SESSIONS 4

typedef struct dt_frame_server_t dt_frame_server_t;

typedef struct dt_webview_shm_t
{
  char session_id[64];
  char shm_names[2][64];
  void *shm_ptr[2];
  size_t shm_size[2];
  uint32_t width;
  uint32_t height;
  int active;
} dt_webview_shm_t;

typedef struct dt_webview_ctx_t
{
  webview_t webview;
  int socket_fd;
  pid_t server_pid;
  char socket_path[PATH_MAX];
  pthread_mutex_t ipc_mutex;          // legacy: used by dt_ipc_request()
  dt_ipc_context_t *ipc_ctx;         // event-aware IPC context (reader thread)
  dt_frame_server_t *frame_server;   // local HTTP server for JPEG frames
  dt_webview_shm_t sessions[DT_WEBVIEW_MAX_SESSIONS];
  pthread_mutex_t session_mutex;
} dt_webview_ctx_t;

// stop the local HTTP frame server
void dt_frame_server_stop(dt_frame_server_t *fs);

// register window drag/zoom bindings (no server needed, safe for splash)
void dt_webview_register_window_bindings(dt_webview_ctx_t *ctx);

// register all JS bindings on the webview (requires server connection)
void dt_webview_register_bindings(dt_webview_ctx_t *ctx);

// shut down the binding worker thread pool
void dt_binding_pool_shutdown(void);
