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
 * Direct (in-process) transport implementation.
 *
 * Routes JSON-RPC calls through the server's dispatch table without
 * any socket I/O. The server handlers call libdarktable directly
 * (introspection-based param access, targeted commit, etc.), so this
 * transport gives us the same functionality as IPC mode but without
 * the serialization overhead of going through a Unix socket.
 *
 * The JSON interface is kept for compatibility with the transport vtable,
 * but the actual param operations happen in-process via direct memory
 * access to module->params (through the server handler code).
 */

#include "transport.h"
#include "server/server.h"

#include <json-glib/json-glib.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

typedef struct dt_direct_transport_data_t
{
  dt_server_t *server;       /* embedded server (no socket) — owned */
  pthread_mutex_t mutex;     /* serializes dispatch calls */
  int next_id;               /* monotonic request ID counter */
} dt_direct_transport_data_t;


/* ── Extract result or error from a JSON-RPC response envelope ──── */

static char *_extract_result(const char *response_json, char **error)
{
  JsonParser *parser = json_parser_new();
  if(!json_parser_load_from_data(parser, response_json, -1, NULL))
  {
    if(error) *error = g_strdup("failed to parse dispatch response");
    g_object_unref(parser);
    return NULL;
  }

  JsonNode *root = json_parser_get_root(parser);
  JsonObject *obj = json_node_get_object(root);

  char *result = NULL;

  if(json_object_has_member(obj, "result"))
  {
    JsonNode *result_node = json_object_get_member(obj, "result");
    JsonGenerator *gen = json_generator_new();
    json_generator_set_root(gen, result_node);
    result = json_generator_to_data(gen, NULL);
    g_object_unref(gen);
  }
  else if(json_object_has_member(obj, "error"))
  {
    JsonObject *err_obj = json_object_get_object_member(obj, "error");
    const char *msg = json_object_get_string_member(err_obj, "message");
    if(error) *error = g_strdup(msg ? msg : "unknown error");
  }
  else
  {
    if(error) *error = g_strdup("response has neither result nor error");
  }

  g_object_unref(parser);
  return result;
}


/* ── call: build request, dispatch, extract result ─────────────── */

static char *_direct_call(dt_webview_transport_t *self,
                          const char *method,
                          const char *params_json,
                          char **error)
{
  dt_direct_transport_data_t *d = self->data;

  fprintf(stderr, "[direct_transport] call: %s(%s)\n", method, params_json ? params_json : "null");

  /* Build a JSON-RPC request string for the server parser */
  char id_str[32];
  int req_id;

  pthread_mutex_lock(&d->mutex);
  req_id = d->next_id++;
  pthread_mutex_unlock(&d->mutex);

  snprintf(id_str, sizeof(id_str), "d%d", req_id);

  char *request_json;
  if(params_json && params_json[0] != '\0')
    request_json = g_strdup_printf(
        "{\"jsonrpc\":\"2.0\",\"id\":\"%s\",\"method\":\"%s\",\"params\":%s}",
        id_str, method, params_json);
  else
    request_json = g_strdup_printf(
        "{\"jsonrpc\":\"2.0\",\"id\":\"%s\",\"method\":\"%s\",\"params\":{}}",
        id_str, method);

  /* Parse into dt_server_request_t */
  dt_server_request_t *req = dt_server_parse_request(request_json, strlen(request_json));
  g_free(request_json);

  if(!req)
  {
    if(error) *error = g_strdup("failed to build request");
    return NULL;
  }

  /* Dispatch through the server's route table (thread-safe: handlers
   * use their own locking where needed) */
  pthread_mutex_lock(&d->mutex);
  char *response = dt_server_dispatch(d->server, req);
  pthread_mutex_unlock(&d->mutex);

  dt_server_free_request(req);

  if(!response)
  {
    if(error) *error = g_strdup("dispatch returned NULL");
    return NULL;
  }

  /* Extract just the result from the JSON-RPC envelope */
  char *result = _extract_result(response, error);
  g_free(response);

  if(result)
    fprintf(stderr, "[direct_transport] result for %s: %.200s%s\n",
            method, result, strlen(result) > 200 ? "..." : "");
  else
    fprintf(stderr, "[direct_transport] error for %s: %s\n",
            method, (error && *error) ? *error : "(null)");

  return result;
}


/* ── get_preview_frame: direct SHM access ─────────────────────── */

static gboolean _direct_get_preview_frame(dt_webview_transport_t *self,
                                          const char *session_id,
                                          dt_transport_frame_t *out_frame)
{
  dt_direct_transport_data_t *d = self->data;
  dt_server_session_t *session = dt_server_find_session(d->server, session_id);
  if(!session) return FALSE;

  int front = session->front_buffer;
  dt_shm_buffer_t *buf = &session->shm_buffers[front];
  if(!buf->mapped || !buf->mapped->ready) return FALSE;

  dt_shm_header_t *hdr = buf->mapped;
  out_frame->pixels = (const uint8_t *)buf->mapped + DT_SHM_HEADER_SIZE;
  out_frame->width = hdr->width;
  out_frame->height = hdr->height;
  out_frame->sequence = hdr->sequence;
  out_frame->owned = FALSE; /* points directly into SHM — zero-copy */

  return TRUE;
}


/* ── set_event_callback ───────────────────────────────────────── */

static void _direct_set_event_callback(dt_webview_transport_t *self,
                                       dt_transport_event_cb callback,
                                       void *user_data)
{
  (void)self;
  (void)callback;
  (void)user_data;
  /* Events are delivered via the embedded server's event_queue.
   * The webview polls the queue in bindings.c event handling.
   * A future improvement can wire this callback to signal handlers. */
}


/* ── destroy ──────────────────────────────────────────────────── */

static void _direct_destroy(dt_webview_transport_t *self)
{
  dt_direct_transport_data_t *d = self->data;

  if(d->server)
  {
    dt_server_cleanup(d->server);
    d->server = NULL;
  }

  pthread_mutex_destroy(&d->mutex);
  g_free(d);
  g_free(self);
}


/* ── Factory ──────────────────────────────────────────────────── */

dt_webview_transport_t *dt_transport_direct_new(void)
{
  /* Create an embedded server (NULL = no socket, calls go directly to handlers) */
  dt_server_t *server = dt_server_init(NULL);
  if(!server)
  {
    fprintf(stderr, "[direct_transport] failed to create embedded server\n");
    return NULL;
  }

  dt_direct_transport_data_t *d = g_new0(dt_direct_transport_data_t, 1);
  d->server = server;
  pthread_mutex_init(&d->mutex, NULL);
  d->next_id = 1;

  dt_webview_transport_t *t = g_new0(dt_webview_transport_t, 1);
  t->data = d;
  t->call = _direct_call;
  t->get_preview_frame = _direct_get_preview_frame;
  t->set_event_callback = _direct_set_event_callback;
  t->destroy = _direct_destroy;

  fprintf(stderr, "[direct_transport] ready (in-process, no IPC)\n");
  return t;
}
