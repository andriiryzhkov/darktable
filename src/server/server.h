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

#include "common/darktable.h"
#include "develop/develop.h"
#include "develop/pixelpipe_hb.h"
#include "server/server_protocol.h"

#include <glib.h>

#define DT_SERVER_MAX_SESSIONS 4

// Forward declarations for handler modules
typedef struct dt_server_t dt_server_t;

// Handler function type: takes server + request params, returns a JSON string response.
// The returned string must be g_free'd by the caller.
typedef char *(*dt_server_handler_t)(dt_server_t *server, const dt_server_request_t *req);

// Route entry in the dispatch table
typedef struct dt_server_route_t
{
  const char *method;
  dt_server_handler_t handler;
} dt_server_route_t;

// Develop session: one open image with its own pipeline
typedef struct dt_server_session_t
{
  char session_id[32];
  dt_imgid_t imgid;

  dt_develop_t dev;
  dt_dev_pixelpipe_t *preview_pipe;

  // Double-buffered shared memory for preview frames
  dt_shm_buffer_t shm_buffers[2];
  int front_buffer;       // index client reads from (0 or 1)
  uint64_t frame_sequence;

  int preview_width;
  int preview_height;
  gboolean dirty;

  // Async pipeline processing (event-driven preview)
  uint64_t pipeline_seq;              // bumped on every set_params
  gboolean pipeline_busy;             // TRUE while worker thread is processing
  pthread_mutex_t pipeline_mutex;
} dt_server_session_t;

struct dt_server_t
{
  // Socket
  int listen_fd;
  int client_fd;
  char socket_path[PATH_MAX];

  // Event loop
  gboolean running;
  GMainLoop *main_loop;

  // Develop sessions
  dt_server_session_t *sessions[DT_SERVER_MAX_SESSIONS];
  int session_count;
  int next_session_id;  // monotonic counter for unique session IDs

  // Event queue (signal callbacks push here, event loop drains to socket)
  GAsyncQueue *event_queue;
};

// Lifecycle
dt_server_t *dt_server_init(const char *socket_path);
void dt_server_run(dt_server_t *server);       // blocking
void dt_server_shutdown(dt_server_t *server);
void dt_server_cleanup(dt_server_t *server);

// Send an event to the connected client (thread-safe via event queue)
void dt_server_queue_event(dt_server_t *server, const char *event_name, JsonNode *data);

// Find a develop session by ID
dt_server_session_t *dt_server_find_session(dt_server_t *server, const char *session_id);

// catalog handlers (dt_server_catalog.c)
char *dt_server_catalog_query(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_get_image(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_get_thumbnail(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_get_thumbnails(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_get_tags(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_get_filmrolls(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_check_imported(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_import(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_copy_import(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_get_file_thumbnail(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_catalog_get_collection_values(dt_server_t *server, const dt_server_request_t *req);

// Develop handlers (dt_server_develop.c)
char *dt_server_develop_open(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_develop_close(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_develop_get_modules(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_develop_get_history(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_develop_get_params(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_develop_set_params(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_develop_commit_params(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_develop_request_preview(dt_server_t *server, const dt_server_request_t *req);
char *dt_server_develop_delete_history(dt_server_t *server, const dt_server_request_t *req);

// Export handlers (dt_server_export.c)
char *dt_server_export_image(dt_server_t *server, const dt_server_request_t *req);

// Event bridge (dt_server_events.c)
void dt_server_events_init(dt_server_t *server);
void dt_server_events_cleanup(dt_server_t *server);
