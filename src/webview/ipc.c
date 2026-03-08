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

#include "ipc.h"
#include "server/server_protocol.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static volatile uint64_t _next_id = 1;

int dt_ipc_connect(const char *socket_path)
{
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if(fd < 0)
  {
    fprintf(stderr, "[webview] socket() failed: %s\n", strerror(errno));
    return -1;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  g_strlcpy(addr.sun_path, socket_path, sizeof(addr.sun_path));

  if(connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
  {
    fprintf(stderr, "[webview] connect(%s) failed: %s\n", socket_path, strerror(errno));
    close(fd);
    return -1;
  }

  fprintf(stderr, "[webview] connected to server: %s\n", socket_path);
  return fd;
}

gboolean dt_ipc_authenticate(int fd, const char *token)
{
  // Send auth handshake: {"id":"auth-0","method":"auth","params":{"token":"..."}}
  char *request = g_strdup_printf(
    "{\"id\":\"auth-0\",\"method\":\"auth\",\"params\":{\"token\":\"%s\"}}", token);

  gboolean ok = dt_server_write_frame(fd, request, strlen(request));
  g_free(request);
  if(!ok)
  {
    fprintf(stderr, "[webview] failed to send auth handshake\n");
    return FALSE;
  }

  // Read auth response
  char *frame = NULL;
  size_t frame_len = 0;
  if(!dt_server_read_frame(fd, &frame, &frame_len))
  {
    fprintf(stderr, "[webview] failed to read auth response\n");
    return FALSE;
  }

  // Check for error
  JsonParser *parser = json_parser_new();
  gboolean auth_ok = FALSE;
  if(json_parser_load_from_data(parser, frame, frame_len, NULL))
  {
    JsonObject *obj = json_node_get_object(json_parser_get_root(parser));
    if(json_object_has_member(obj, "error") && !json_object_get_null_member(obj, "error"))
    {
      JsonObject *err = json_object_get_object_member(obj, "error");
      fprintf(stderr, "[webview] auth failed: %s\n",
              json_object_get_string_member(err, "message"));
    }
    else
    {
      auth_ok = TRUE;
      fprintf(stderr, "[webview] authenticated successfully\n");
    }
  }
  g_free(frame);
  g_object_unref(parser);
  return auth_ok;
}

char *dt_ipc_request(int fd, pthread_mutex_t *mutex,
                     const char *method, const char *params_json,
                     char **out_error)
{
  if(out_error) *out_error = NULL;

  // Build request JSON: {"id":"req-N","method":"...","params":{...}}
  uint64_t id = __atomic_fetch_add(&_next_id, 1, __ATOMIC_RELAXED);
  char id_str[32];
  snprintf(id_str, sizeof(id_str), "req-%" PRIu64, id);

  char *request;
  if(params_json && params_json[0])
    request = g_strdup_printf("{\"id\":\"%s\",\"method\":\"%s\",\"params\":%s}",
                              id_str, method, params_json);
  else
    request = g_strdup_printf("{\"id\":\"%s\",\"method\":\"%s\",\"params\":{}}",
                              id_str, method);

  char *result_json = NULL;

  pthread_mutex_lock(mutex);

  // Send request frame
  if(!dt_server_write_frame(fd, request, strlen(request)))
  {
    if(out_error) *out_error = g_strdup("failed to write request frame");
    g_free(request);
    pthread_mutex_unlock(mutex);
    return NULL;
  }
  g_free(request);

  // Read response(s), skipping events (id=null).
  // Limit event skips to prevent infinite loop if server only sends events.
  const int max_event_skips = 1000;
  int events_skipped = 0;
  while(TRUE)
  {
    char *frame = NULL;
    size_t frame_len = 0;
    if(!dt_server_read_frame(fd, &frame, &frame_len))
    {
      if(out_error) *out_error = g_strdup("failed to read response frame");
      pthread_mutex_unlock(mutex);
      return NULL;
    }

    // Parse JSON
    JsonParser *parser = json_parser_new();
    if(!json_parser_load_from_data(parser, frame, frame_len, NULL))
    {
      if(out_error) *out_error = g_strdup("failed to parse response JSON");
      g_free(frame);
      g_object_unref(parser);
      pthread_mutex_unlock(mutex);
      return NULL;
    }
    g_free(frame);

    JsonNode *root = json_parser_get_root(parser);
    JsonObject *obj = json_node_get_object(root);

    // Skip events (id is null or missing, has "event" field)
    if(json_object_has_member(obj, "event"))
    {
      fprintf(stderr, "[webview] skipping event: %s\n",
              json_object_get_string_member(obj, "event"));
      g_object_unref(parser);
      if(++events_skipped >= max_event_skips)
      {
        if(out_error) *out_error = g_strdup("too many events without a response");
        pthread_mutex_unlock(mutex);
        return NULL;
      }
      continue;
    }

    // Check for error
    if(json_object_has_member(obj, "error") && !json_object_get_null_member(obj, "error"))
    {
      JsonObject *err = json_object_get_object_member(obj, "error");
      const char *msg = json_object_get_string_member(err, "message");
      if(out_error) *out_error = g_strdup(msg ? msg : "unknown server error");
      g_object_unref(parser);
      pthread_mutex_unlock(mutex);
      return NULL;
    }

    // Extract result
    if(json_object_has_member(obj, "result") && !json_object_get_null_member(obj, "result"))
    {
      JsonNode *result_node = json_object_get_member(obj, "result");
      JsonGenerator *gen = json_generator_new();
      json_generator_set_root(gen, result_node);
      result_json = json_generator_to_data(gen, NULL);
      g_object_unref(gen);
    }
    else
    {
      result_json = g_strdup("null");
    }

    g_object_unref(parser);
    break;
  }

  pthread_mutex_unlock(mutex);
  return result_json;
}

// --- Event-aware IPC context (Phase 1A) ---

#define DT_IPC_MAX_PENDING 32

typedef struct dt_ipc_pending_t
{
  char id_str[32];
  char *result_json;
  char *error_msg;
  gboolean completed;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} dt_ipc_pending_t;

struct dt_ipc_context_t
{
  int fd;
  pthread_t reader_thread;
  gboolean running;

  // Write serialization (multiple callers may send requests concurrently)
  pthread_mutex_t write_mutex;

  // Pending request slots
  dt_ipc_pending_t *pending[DT_IPC_MAX_PENDING];
  pthread_mutex_t pending_mutex;

  // Event callback
  dt_ipc_event_cb_t event_cb;
  void *event_user_data;
};

static dt_ipc_pending_t *_pending_new(const char *id_str)
{
  dt_ipc_pending_t *p = g_new0(dt_ipc_pending_t, 1);
  g_strlcpy(p->id_str, id_str, sizeof(p->id_str));
  p->result_json = NULL;
  p->error_msg = NULL;
  p->completed = FALSE;
  pthread_mutex_init(&p->mutex, NULL);
  pthread_cond_init(&p->cond, NULL);
  return p;
}

static void _pending_free(dt_ipc_pending_t *p)
{
  pthread_mutex_destroy(&p->mutex);
  pthread_cond_destroy(&p->cond);
  g_free(p->result_json);
  g_free(p->error_msg);
  g_free(p);
}

// Register a pending slot.  Returns the slot index or -1 on overflow.
static int _register_pending(dt_ipc_context_t *ctx, dt_ipc_pending_t *p)
{
  pthread_mutex_lock(&ctx->pending_mutex);
  for(int i = 0; i < DT_IPC_MAX_PENDING; i++)
  {
    if(!ctx->pending[i])
    {
      ctx->pending[i] = p;
      pthread_mutex_unlock(&ctx->pending_mutex);
      return i;
    }
  }
  pthread_mutex_unlock(&ctx->pending_mutex);
  return -1;
}

static void _unregister_pending(dt_ipc_context_t *ctx, int idx)
{
  pthread_mutex_lock(&ctx->pending_mutex);
  ctx->pending[idx] = NULL;
  pthread_mutex_unlock(&ctx->pending_mutex);
}

// Extract "result" or "error" from a parsed JSON-RPC response and fill the pending slot.
static void _complete_pending(dt_ipc_pending_t *p, JsonObject *obj)
{
  pthread_mutex_lock(&p->mutex);

  if(json_object_has_member(obj, "error") && !json_object_get_null_member(obj, "error"))
  {
    JsonObject *err = json_object_get_object_member(obj, "error");
    const char *msg = json_object_get_string_member(err, "message");
    p->error_msg = g_strdup(msg ? msg : "unknown server error");
  }
  else if(json_object_has_member(obj, "result") && !json_object_get_null_member(obj, "result"))
  {
    JsonNode *result_node = json_object_get_member(obj, "result");
    JsonGenerator *gen = json_generator_new();
    json_generator_set_root(gen, result_node);
    p->result_json = json_generator_to_data(gen, NULL);
    g_object_unref(gen);
  }
  else
  {
    p->result_json = g_strdup("null");
  }

  p->completed = TRUE;
  pthread_cond_signal(&p->cond);
  pthread_mutex_unlock(&p->mutex);
}

static void *_reader_thread_fn(void *arg)
{
  dt_ipc_context_t *ctx = arg;

  while(ctx->running)
  {
    char *frame = NULL;
    size_t frame_len = 0;
    if(!dt_server_read_frame(ctx->fd, &frame, &frame_len))
    {
      if(ctx->running)
        fprintf(stderr, "[webview] IPC reader: read_frame failed, exiting\n");
      break;
    }

    JsonParser *parser = json_parser_new();
    if(!json_parser_load_from_data(parser, frame, frame_len, NULL))
    {
      fprintf(stderr, "[webview] IPC reader: failed to parse JSON\n");
      g_free(frame);
      g_object_unref(parser);
      continue;
    }
    g_free(frame);

    JsonNode *root = json_parser_get_root(parser);
    JsonObject *obj = json_node_get_object(root);

    // Server-pushed event?
    if(json_object_has_member(obj, "event"))
    {
      const char *event_name = json_object_get_string_member(obj, "event");
      if(ctx->event_cb && event_name)
      {
        // Serialize the "data" field as JSON string for the callback
        char *data_json = NULL;
        if(json_object_has_member(obj, "data"))
        {
          JsonNode *params_node = json_object_get_member(obj, "data");
          JsonGenerator *gen = json_generator_new();
          json_generator_set_root(gen, params_node);
          data_json = json_generator_to_data(gen, NULL);
          g_object_unref(gen);
        }
        ctx->event_cb(event_name, data_json ? data_json : "{}", ctx->event_user_data);
        g_free(data_json);
      }
      g_object_unref(parser);
      continue;
    }

    // RPC response — match by "id"
    if(json_object_has_member(obj, "id") && !json_object_get_null_member(obj, "id"))
    {
      const char *resp_id = json_object_get_string_member(obj, "id");
      if(resp_id)
      {
        pthread_mutex_lock(&ctx->pending_mutex);
        for(int i = 0; i < DT_IPC_MAX_PENDING; i++)
        {
          if(ctx->pending[i] && !strcmp(ctx->pending[i]->id_str, resp_id))
          {
            _complete_pending(ctx->pending[i], obj);
            break;
          }
        }
        pthread_mutex_unlock(&ctx->pending_mutex);
      }
    }

    g_object_unref(parser);
  }

  // Wake up any pending requests so they don't hang
  pthread_mutex_lock(&ctx->pending_mutex);
  for(int i = 0; i < DT_IPC_MAX_PENDING; i++)
  {
    if(ctx->pending[i])
    {
      pthread_mutex_lock(&ctx->pending[i]->mutex);
      if(!ctx->pending[i]->completed)
      {
        ctx->pending[i]->error_msg = g_strdup("IPC connection closed");
        ctx->pending[i]->completed = TRUE;
        pthread_cond_signal(&ctx->pending[i]->cond);
      }
      pthread_mutex_unlock(&ctx->pending[i]->mutex);
    }
  }
  pthread_mutex_unlock(&ctx->pending_mutex);

  return NULL;
}

dt_ipc_context_t *dt_ipc_context_new(int fd, dt_ipc_event_cb_t event_cb, void *event_user_data)
{
  dt_ipc_context_t *ctx = g_new0(dt_ipc_context_t, 1);
  ctx->fd = fd;
  ctx->running = TRUE;
  ctx->event_cb = event_cb;
  ctx->event_user_data = event_user_data;
  pthread_mutex_init(&ctx->write_mutex, NULL);
  pthread_mutex_init(&ctx->pending_mutex, NULL);
  memset(ctx->pending, 0, sizeof(ctx->pending));

  if(pthread_create(&ctx->reader_thread, NULL, _reader_thread_fn, ctx) != 0)
  {
    fprintf(stderr, "[webview] failed to create IPC reader thread: %s\n", strerror(errno));
    pthread_mutex_destroy(&ctx->write_mutex);
    pthread_mutex_destroy(&ctx->pending_mutex);
    g_free(ctx);
    return NULL;
  }

  fprintf(stderr, "[webview] IPC context created with reader thread\n");
  return ctx;
}

void dt_ipc_context_free(dt_ipc_context_t *ctx)
{
  if(!ctx) return;

  ctx->running = FALSE;

  // Shut down the socket to unblock the reader thread's read_frame
  shutdown(ctx->fd, SHUT_RDWR);

  pthread_join(ctx->reader_thread, NULL);

  pthread_mutex_destroy(&ctx->write_mutex);
  pthread_mutex_destroy(&ctx->pending_mutex);
  g_free(ctx);
}

char *dt_ipc_request2(dt_ipc_context_t *ctx,
                      const char *method, const char *params_json,
                      char **out_error)
{
  if(out_error) *out_error = NULL;

  // Build request JSON
  uint64_t id = __atomic_fetch_add(&_next_id, 1, __ATOMIC_RELAXED);
  char id_str[32];
  snprintf(id_str, sizeof(id_str), "req-%" PRIu64, id);

  char *request;
  if(params_json && params_json[0])
    request = g_strdup_printf("{\"id\":\"%s\",\"method\":\"%s\",\"params\":%s}",
                              id_str, method, params_json);
  else
    request = g_strdup_printf("{\"id\":\"%s\",\"method\":\"%s\",\"params\":{}}",
                              id_str, method);

  // Allocate a pending slot
  dt_ipc_pending_t *p = _pending_new(id_str);
  int slot = _register_pending(ctx, p);
  if(slot < 0)
  {
    if(out_error) *out_error = g_strdup("too many pending IPC requests");
    g_free(request);
    _pending_free(p);
    return NULL;
  }

  // Send the request frame (serialized with other writers)
  pthread_mutex_lock(&ctx->write_mutex);
  gboolean written = dt_server_write_frame(ctx->fd, request, strlen(request));
  pthread_mutex_unlock(&ctx->write_mutex);
  g_free(request);

  if(!written)
  {
    _unregister_pending(ctx, slot);
    _pending_free(p);
    if(out_error) *out_error = g_strdup("failed to write request frame");
    return NULL;
  }

  // Wait for the reader thread to complete this slot
  pthread_mutex_lock(&p->mutex);
  while(!p->completed)
    pthread_cond_wait(&p->cond, &p->mutex);
  pthread_mutex_unlock(&p->mutex);

  _unregister_pending(ctx, slot);

  // Extract result
  char *result = p->result_json;
  p->result_json = NULL; // transfer ownership

  if(!result && p->error_msg)
  {
    if(out_error) *out_error = g_strdup(p->error_msg);
  }

  _pending_free(p);
  return result;
}
