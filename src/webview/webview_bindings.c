#include "webview_bindings.h"
#include "ipc_client.h"
#include "server/server_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <json-glib/json-glib.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// ── async callback infrastructure ───────────────────────────────────────────

typedef struct async_req_t
{
  dt_webview_ctx_t *ctx;
  char *id;
  char *req;
} async_req_t;

static async_req_t *_async_req_new(dt_webview_ctx_t *ctx, const char *id, const char *req)
{
  async_req_t *ar = g_new0(async_req_t, 1);
  ar->ctx = ctx;
  ar->id = g_strdup(id);
  ar->req = g_strdup(req);
  return ar;
}

static void _async_req_free(async_req_t *ar)
{
  g_free(ar->id);
  g_free(ar->req);
  g_free(ar);
}

static void _return_ok(dt_webview_ctx_t *ctx, const char *id, const char *json_result)
{
  webview_return(ctx->webview, id, 0, json_result ? json_result : "null");
}

static void _return_error(dt_webview_ctx_t *ctx, const char *id, const char *msg)
{
  // Escape the message for JSON string
  char *escaped = g_strescape(msg, NULL);
  char *json = g_strdup_printf("\"%s\"", escaped);
  webview_return(ctx->webview, id, 1, json);
  g_free(json);
  g_free(escaped);
}

// Helper: parse JSON array string into a JsonArray.
// Returns NULL on error. Caller must unref the parser.
static JsonArray *_parse_args(const char *req, JsonParser **out_parser)
{
  *out_parser = json_parser_new();
  if(!json_parser_load_from_data(*out_parser, req, -1, NULL))
  {
    g_object_unref(*out_parser);
    *out_parser = NULL;
    return NULL;
  }
  JsonNode *root = json_parser_get_root(*out_parser);
  if(!root || !JSON_NODE_HOLDS_ARRAY(root))
  {
    g_object_unref(*out_parser);
    *out_parser = NULL;
    return NULL;
  }
  return json_node_get_array(root);
}

// ── IPC passthrough helper ──────────────────────────────────────────────────

// Generic worker: send method+params via IPC, return result to JS
typedef struct ipc_passthrough_t
{
  dt_webview_ctx_t *ctx;
  char *id;
  char *method;
  char *params_json;
} ipc_passthrough_t;

static void *_ipc_passthrough_worker(void *arg)
{
  ipc_passthrough_t *pt = arg;
  char *error = NULL;
  fprintf(stderr, "[webview] IPC request: %s(%s)\n", pt->method, pt->params_json);
  char *result = dt_ipc_request(pt->ctx->socket_fd, &pt->ctx->ipc_mutex,
                                pt->method, pt->params_json, &error);
  if(result)
  {
    // Log truncated result for debugging
    fprintf(stderr, "[webview] IPC result for %s: %.200s%s\n",
            pt->method, result, strlen(result) > 200 ? "..." : "");
    _return_ok(pt->ctx, pt->id, result);
  }
  else
  {
    fprintf(stderr, "[webview] IPC error for %s: %s\n", pt->method, error ? error : "(null)");
    _return_error(pt->ctx, pt->id, error ? error : "IPC request failed");
  }

  g_free(result);
  g_free(error);
  g_free(pt->id);
  g_free(pt->method);
  g_free(pt->params_json);
  g_free(pt);
  return NULL;
}

static void _ipc_passthrough(dt_webview_ctx_t *ctx, const char *id,
                              const char *method, const char *params_json)
{
  ipc_passthrough_t *pt = g_new0(ipc_passthrough_t, 1);
  pt->ctx = ctx;
  pt->id = g_strdup(id);
  pt->method = g_strdup(method);
  pt->params_json = g_strdup(params_json ? params_json : "{}");

  pthread_t thread;
  pthread_create(&thread, NULL, _ipc_passthrough_worker, pt);
  pthread_detach(thread);
}

// ── ping ────────────────────────────────────────────────────────────────────

static void on_ping(const char *id, const char *req, void *arg)
{
  (void)req;
  _ipc_passthrough(arg, id, "system.ping", "{}");
}

// ── catalog.query ───────────────────────────────────────────────────────────

static void on_catalog_query(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "catalogQuery requires (offset, limit)");
    if(parser) g_object_unref(parser);
    return;
  }

  gint64 offset = json_array_get_int_element(args, 0);
  gint64 limit = json_array_get_int_element(args, 1);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"offset\":%" G_GINT64_FORMAT ",\"limit\":%" G_GINT64_FORMAT "}", offset, limit);
  _ipc_passthrough(ctx, id, "catalog.query", params);
  g_free(params);
}

// ── catalog.get_thumbnail ───────────────────────────────────────────────────

static void on_catalog_get_thumbnail(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "catalogGetThumbnail requires (imgid)");
    if(parser) g_object_unref(parser);
    return;
  }

  gint64 imgid = json_array_get_int_element(args, 0);
  g_object_unref(parser);

  char *params = g_strdup_printf("{\"imgid\":%" G_GINT64_FORMAT "}", imgid);
  _ipc_passthrough(ctx, id, "catalog.get_thumbnail", params);
  g_free(params);
}

// ── develop.open ────────────────────────────────────────────────────────────

static void *_develop_open_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  // Parse args: [imgid, width, height]
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, ar->id, "developOpen requires (imgid, width, height)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  gint64 imgid = json_array_get_int_element(args, 0);
  gint64 width = json_array_get_int_element(args, 1);
  gint64 height = json_array_get_int_element(args, 2);
  g_object_unref(parser);

  // IPC request
  char *params = g_strdup_printf("{\"imgid\":%" G_GINT64_FORMAT
                                  ",\"width\":%" G_GINT64_FORMAT
                                  ",\"height\":%" G_GINT64_FORMAT "}", imgid, width, height);
  char *error = NULL;
  char *result = dt_ipc_request(ctx->socket_fd, &ctx->ipc_mutex,
                                "develop.open", params, &error);
  g_free(params);

  if(!result)
  {
    _return_error(ctx, ar->id, error ? error : "develop.open failed");
    g_free(error);
    _async_req_free(ar);
    return NULL;
  }

  // Parse result to get shm_names and open SHM handles
  JsonParser *rparser = json_parser_new();
  if(json_parser_load_from_data(rparser, result, -1, NULL))
  {
    JsonObject *robj = json_node_get_object(json_parser_get_root(rparser));
    const char *session_id = json_object_get_string_member(robj, "session_id");
    JsonArray *shm_names = json_object_get_array_member(robj, "shm_names");
    gint64 pw = json_object_get_int_member(robj, "preview_width");
    gint64 ph = json_object_get_int_member(robj, "preview_height");

    if(session_id && shm_names && json_array_get_length(shm_names) >= 2)
    {
      const char *name0 = json_array_get_string_element(shm_names, 0);
      const char *name1 = json_array_get_string_element(shm_names, 1);

      size_t shm_size = DT_SHM_HEADER_SIZE + (size_t)pw * (size_t)ph * 4;

      pthread_mutex_lock(&ctx->session_mutex);
      // Find empty slot
      int slot = -1;
      for(int i = 0; i < DT_WEBVIEW_MAX_SESSIONS; i++)
      {
        if(!ctx->sessions[i].active)
        {
          slot = i;
          break;
        }
      }
      if(slot >= 0)
      {
        dt_webview_shm_t *s = &ctx->sessions[slot];
        g_strlcpy(s->session_id, session_id, sizeof(s->session_id));
        g_strlcpy(s->shm_names[0], name0, sizeof(s->shm_names[0]));
        g_strlcpy(s->shm_names[1], name1, sizeof(s->shm_names[1]));
        s->width = (uint32_t)pw;
        s->height = (uint32_t)ph;
        s->shm_size[0] = shm_size;
        s->shm_size[1] = shm_size;

        // Open SHM read-only
        for(int b = 0; b < 2; b++)
        {
          int fd = shm_open(s->shm_names[b], O_RDONLY, 0);
          if(fd >= 0)
          {
            s->shm_ptr[b] = mmap(NULL, shm_size, PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if(s->shm_ptr[b] == MAP_FAILED)
            {
              fprintf(stderr, "[webview] mmap(%s) failed: %s\n", s->shm_names[b], strerror(errno));
              s->shm_ptr[b] = NULL;
            }
          }
          else
          {
            fprintf(stderr, "[webview] shm_open(%s) failed: %s\n", s->shm_names[b], strerror(errno));
            s->shm_ptr[b] = NULL;
          }
        }
        s->active = 1;
      }
      pthread_mutex_unlock(&ctx->session_mutex);
    }
  }
  g_object_unref(rparser);

  _return_ok(ctx, ar->id, result);
  g_free(result);
  _async_req_free(ar);
  return NULL;
}

static void on_develop_open(const char *id, const char *req, void *arg)
{
  async_req_t *ar = _async_req_new(arg, id, req);
  pthread_t thread;
  pthread_create(&thread, NULL, _develop_open_worker, ar);
  pthread_detach(thread);
}

// ── develop.close ───────────────────────────────────────────────────────────

static void *_develop_close_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, ar->id, "developClose requires (sessionId)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  const char *session_id = json_array_get_string_element(args, 0);

  // Close SHM handles
  pthread_mutex_lock(&ctx->session_mutex);
  for(int i = 0; i < DT_WEBVIEW_MAX_SESSIONS; i++)
  {
    if(ctx->sessions[i].active && !strcmp(ctx->sessions[i].session_id, session_id))
    {
      for(int b = 0; b < 2; b++)
      {
        if(ctx->sessions[i].shm_ptr[b])
        {
          munmap(ctx->sessions[i].shm_ptr[b], ctx->sessions[i].shm_size[b]);
          ctx->sessions[i].shm_ptr[b] = NULL;
        }
      }
      ctx->sessions[i].active = 0;
      break;
    }
  }
  pthread_mutex_unlock(&ctx->session_mutex);

  // IPC close
  char *params = g_strdup_printf("{\"session_id\":\"%s\"}", session_id);
  g_object_unref(parser);

  char *error = NULL;
  char *result = dt_ipc_request(ctx->socket_fd, &ctx->ipc_mutex,
                                "develop.close", params, &error);
  g_free(params);

  if(result)
    _return_ok(ctx, ar->id, result);
  else
    _return_error(ctx, ar->id, error ? error : "develop.close failed");

  g_free(result);
  g_free(error);
  _async_req_free(ar);
  return NULL;
}

static void on_develop_close(const char *id, const char *req, void *arg)
{
  async_req_t *ar = _async_req_new(arg, id, req);
  pthread_t thread;
  pthread_create(&thread, NULL, _develop_close_worker, ar);
  pthread_detach(thread);
}

// ── develop.set_params ──────────────────────────────────────────────────────

static void on_develop_set_params(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, id, "developSetParams requires (sessionId, op, params)");
    if(parser) g_object_unref(parser);
    return;
  }

  const char *session_id = json_array_get_string_element(args, 0);
  const char *op = json_array_get_string_element(args, 1);

  // Serialize the params object back to JSON string
  JsonNode *params_node = json_array_get_element(args, 2);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, params_node);
  char *params_str = json_generator_to_data(gen, NULL);
  g_object_unref(gen);

  char *ipc_params = g_strdup_printf("{\"session_id\":\"%s\",\"op\":\"%s\",\"params\":%s}",
                                     session_id, op, params_str);
  g_free(params_str);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, "develop.set_params", ipc_params);
  g_free(ipc_params);
}

// ── develop.request_preview ─────────────────────────────────────────────────

static void on_develop_request_preview(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "developRequestPreview requires (sessionId)");
    if(parser) g_object_unref(parser);
    return;
  }

  const char *session_id = json_array_get_string_element(args, 0);
  char *params = g_strdup_printf("{\"session_id\":\"%s\"}", session_id);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, "develop.request_preview", params);
  g_free(params);
}

// ── getPreviewFrame (SHM direct read, no IPC) ──────────────────────────────

static void *_get_preview_frame_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, ar->id, "getPreviewFrame requires (sessionId, frontBuffer)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  const char *session_id = json_array_get_string_element(args, 0);
  gint64 front_buffer = json_array_get_int_element(args, 1);
  g_object_unref(parser);

  // Find session
  pthread_mutex_lock(&ctx->session_mutex);
  dt_webview_shm_t *session = NULL;
  for(int i = 0; i < DT_WEBVIEW_MAX_SESSIONS; i++)
  {
    if(ctx->sessions[i].active && !strcmp(ctx->sessions[i].session_id, session_id))
    {
      session = &ctx->sessions[i];
      break;
    }
  }

  if(!session)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _return_error(ctx, ar->id, "session not found");
    _async_req_free(ar);
    return NULL;
  }

  int buf_idx = (front_buffer == 0) ? 0 : 1;
  void *ptr = session->shm_ptr[buf_idx];

  if(!ptr)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _return_error(ctx, ar->id, "SHM buffer not mapped");
    _async_req_free(ar);
    return NULL;
  }

  // Validate SHM header
  dt_shm_header_t *header = (dt_shm_header_t *)ptr;
  if(header->magic != DT_SHM_MAGIC || header->version != DT_SHM_VERSION)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _return_error(ctx, ar->id, "invalid SHM header");
    _async_req_free(ar);
    return NULL;
  }

  uint32_t ready = __atomic_load_n(&header->ready, __ATOMIC_ACQUIRE);
  if(ready != 1)
  {
    pthread_mutex_unlock(&ctx->session_mutex);
    _return_error(ctx, ar->id, "frame not ready");
    _async_req_free(ar);
    return NULL;
  }

  uint32_t w = header->width;
  uint32_t h = header->height;
  size_t pixel_size = (size_t)w * (size_t)h * 4;
  uint8_t *pixels = (uint8_t *)ptr + DT_SHM_HEADER_SIZE;

  // Base64 encode pixels
  gchar *b64 = g_base64_encode(pixels, pixel_size);
  pthread_mutex_unlock(&ctx->session_mutex);

  // Build JSON result: {"width":W, "height":H, "data":"base64..."}
  char *result = g_strdup_printf("{\"width\":%u,\"height\":%u,\"data\":\"%s\"}", w, h, b64);
  g_free(b64);

  _return_ok(ctx, ar->id, result);
  g_free(result);
  _async_req_free(ar);
  return NULL;
}

static void on_get_preview_frame(const char *id, const char *req, void *arg)
{
  async_req_t *ar = _async_req_new(arg, id, req);
  pthread_t thread;
  pthread_create(&thread, NULL, _get_preview_frame_worker, ar);
  pthread_detach(thread);
}

// ── Registration ────────────────────────────────────────────────────────────

void dt_webview_register_bindings(dt_webview_ctx_t *ctx)
{
  webview_bind(ctx->webview, "ping", on_ping, ctx);
  webview_bind(ctx->webview, "catalogQuery", on_catalog_query, ctx);
  webview_bind(ctx->webview, "catalogGetThumbnail", on_catalog_get_thumbnail, ctx);
  webview_bind(ctx->webview, "developOpen", on_develop_open, ctx);
  webview_bind(ctx->webview, "developClose", on_develop_close, ctx);
  webview_bind(ctx->webview, "developSetParams", on_develop_set_params, ctx);
  webview_bind(ctx->webview, "developRequestPreview", on_develop_request_preview, ctx);
  webview_bind(ctx->webview, "getPreviewFrame", on_get_preview_frame, ctx);
}
