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
 * Owns the socket, IPC context, and server process lifecycle.
 * shutdown() handles graceful server termination.
 */

#include "transport.h"
#include "ipc.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct dt_ipc_transport_data_t
{
  int socket_fd;                  /* connected Unix socket (owned) */
  pthread_mutex_t legacy_mutex;   /* mutex for legacy dt_ipc_request() */
  dt_ipc_context_t *ipc_ctx;     /* event-aware IPC context (owned) */
  pid_t server_pid;               /* server process PID (owned, -1 if none) */
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


/* ── shutdown: graceful server termination ──────────────────────── */

static void _ipc_shutdown(dt_webview_transport_t *self)
{
  dt_ipc_transport_data_t *d = self->data;

  /* Send graceful shutdown RPC so the server saves config */
  if(d->socket_fd >= 0)
  {
    fprintf(stderr, "[ipc_transport] sending system.shutdown to server\n");
    char *error = NULL;
    char *resp = self->call(self, "system.shutdown", "{}", &error);
    g_free(resp);
    g_free(error);
  }

  /* Stop IPC reader thread */
  if(d->ipc_ctx)
  {
    dt_ipc_context_free(d->ipc_ctx);
    d->ipc_ctx = NULL;
  }

  /* Close socket */
  if(d->socket_fd >= 0)
  {
    close(d->socket_fd);
    d->socket_fd = -1;
  }

  /* Wait for server to exit cleanly, force kill if needed */
  if(d->server_pid > 0)
  {
    int status;
    for(int i = 0; i < 30; i++)
    {
      pid_t ret = waitpid(d->server_pid, &status, WNOHANG);
      if(ret != 0) goto server_done;
      struct timespec ts = { .tv_sec = 0, .tv_nsec = 100000000 }; // 100ms
      nanosleep(&ts, NULL);
    }
    fprintf(stderr, "[ipc_transport] server did not exit, sending SIGTERM\n");
    kill(d->server_pid, SIGTERM);
    waitpid(d->server_pid, &status, 0);
server_done:
    fprintf(stderr, "[ipc_transport] server stopped\n");
    d->server_pid = -1;
  }
}


/* ── destroy ────────────────────────────────────────────────────── */

static void _ipc_destroy(dt_webview_transport_t *self)
{
  dt_ipc_transport_data_t *d = self->data;

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
  d->server_pid = -1;
  pthread_mutex_init(&d->legacy_mutex, NULL);
  d->ipc_ctx = NULL; /* caller sets up via dt_transport_ipc_set_context() */

  dt_webview_transport_t *t = g_new0(dt_webview_transport_t, 1);
  t->data = d;
  t->call = _ipc_call;
  t->get_preview_frame = _ipc_get_preview_frame;
  t->set_event_callback = _ipc_set_event_callback;
  t->shutdown = _ipc_shutdown;
  t->destroy = _ipc_destroy;
  return t;
}


/* ── Accessors ──────────────────────────────────────────────────── */

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

void dt_transport_ipc_set_server_pid(dt_webview_transport_t *t, pid_t pid)
{
  dt_ipc_transport_data_t *d = t->data;
  d->server_pid = pid;
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
