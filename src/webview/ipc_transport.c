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

/*
 * IPC transport implementation of the transport vtable.
 *
 * Wraps the existing dt_ipc_request2() / dt_ipc_request() dual-path
 * dispatch pattern behind the dt_webview_transport_t interface.
 *
 * This is a pure refactoring — the IPC behavior is identical to what
 * bindings.c did before, just moved behind the vtable.
 */

#include "transport.h"
#include "ipc.h"

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

typedef struct dt_ipc_transport_data_t
{
  int socket_fd;                  /* connected Unix socket (NOT owned — main.c manages lifecycle) */
  pthread_mutex_t legacy_mutex;   /* mutex for legacy dt_ipc_request() */
  dt_ipc_context_t *ipc_ctx;     /* event-aware IPC context (NOT owned — main.c manages lifecycle) */
} dt_ipc_transport_data_t;


/* ── call: JSON-RPC request via IPC ─────────────────────────────── */

static char *_ipc_call(dt_webview_transport_t *self,
                       const char *method,
                       const char *params_json,
                       char **error)
{
  dt_ipc_transport_data_t *d = self->data;

  fprintf(stderr, "[ipc_transport] call: %s(%s)\n", method, params_json ? params_json : "null");

  char *result;
  if(d->ipc_ctx)
    result = dt_ipc_request2(d->ipc_ctx, method, params_json, error);
  else
    result = dt_ipc_request(d->socket_fd, &d->legacy_mutex, method, params_json, error);

  if(result)
    fprintf(stderr, "[ipc_transport] result for %s: %.200s%s\n",
            method, result, strlen(result) > 200 ? "..." : "");
  else
    fprintf(stderr, "[ipc_transport] error for %s: %s\n",
            method, (error && *error) ? *error : "(null)");

  return result;
}


/* ── get_preview_frame: stub (SHM access stays in bindings.c) ───── */

static gboolean _ipc_get_preview_frame(dt_webview_transport_t *self,
                                       const char *session_id,
                                       dt_transport_frame_t *out_frame)
{
  (void)self;
  (void)session_id;
  (void)out_frame;
  /* Preview frame access is handled directly by bindings.c via SHM.
   * The SHM lifecycle (open/close/read) is tightly coupled to the
   * develop session management in bindings.c, so it stays there
   * for now. A future refactoring can move it here. */
  return FALSE;
}


/* ── set_event_callback: delegate to IPC context ────────────────── */

static void _ipc_set_event_callback(dt_webview_transport_t *self,
                                    dt_transport_event_cb callback,
                                    void *user_data)
{
  (void)self;
  (void)callback;
  (void)user_data;
  /* Event callbacks are wired up during dt_ipc_context_new() in main.c.
   * The transport doesn't own event subscription lifecycle yet.
   * A future refactoring can move this here. */
}


/* ── destroy ────────────────────────────────────────────────────── */

static void _ipc_destroy(dt_webview_transport_t *self)
{
  dt_ipc_transport_data_t *d = self->data;

  /* NOTE: socket_fd and ipc_ctx are NOT owned by the transport.
   * main.c manages their lifecycle (ipc_context_free, close).
   * We only clean up our own resources. */
  d->ipc_ctx = NULL;
  d->socket_fd = -1;

  pthread_mutex_destroy(&d->legacy_mutex);
  g_free(d);
  g_free(self);
}


/* ── Factory ────────────────────────────────────────────────────── */

dt_webview_transport_t *dt_transport_ipc_new(int socket_fd)
{
  if(socket_fd < 0) return NULL;

  dt_ipc_transport_data_t *d = g_new0(dt_ipc_transport_data_t, 1);
  d->socket_fd = socket_fd;
  pthread_mutex_init(&d->legacy_mutex, NULL);
  d->ipc_ctx = NULL; /* caller sets up via dt_transport_ipc_set_context() */

  dt_webview_transport_t *t = g_new0(dt_webview_transport_t, 1);
  t->data = d;
  t->call = _ipc_call;
  t->get_preview_frame = _ipc_get_preview_frame;
  t->set_event_callback = _ipc_set_event_callback;
  t->destroy = _ipc_destroy;
  return t;
}


/* ── Accessor for wiring up the event-aware IPC context ─────────── */

void dt_transport_ipc_set_context(dt_webview_transport_t *t,
                                  dt_ipc_context_t *ipc_ctx)
{
  dt_ipc_transport_data_t *d = t->data;
  d->ipc_ctx = ipc_ctx;
}

dt_ipc_context_t *dt_transport_ipc_get_context(dt_webview_transport_t *t)
{
  dt_ipc_transport_data_t *d = t->data;
  return d->ipc_ctx;
}

int dt_transport_ipc_get_fd(dt_webview_transport_t *t)
{
  dt_ipc_transport_data_t *d = t->data;
  return d->socket_fd;
}

pthread_mutex_t *dt_transport_ipc_get_mutex(dt_webview_transport_t *t)
{
  dt_ipc_transport_data_t *d = t->data;
  return &d->legacy_mutex;
}
