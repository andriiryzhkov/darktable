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

#include "bindings.h"
#include "ipc.h"
#include "titlebar.h"
#include "server/server_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <glib/gstdio.h>
#include <json-glib/json-glib.h>
#include <nfd.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/mman.h>
#endif
#include <unistd.h>

/* async callback infrastructure */

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
  // Use json-glib to properly escape the message for JSON
  JsonNode *node = json_node_new(JSON_NODE_VALUE);
  json_node_set_string(node, msg);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, node);
  char *json = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(node);
  webview_return(ctx->webview, id, 1, json);
  g_free(json);
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

/* IPC passthrough: send method+params via IPC, return result to JS */
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
  if(pthread_create(&thread, NULL, _ipc_passthrough_worker, pt) != 0)
  {
    fprintf(stderr, "[webview] pthread_create failed for %s: %s\n", method, strerror(errno));
    _return_error(ctx, id, "internal error: failed to create worker thread");
    g_free(pt->id);
    g_free(pt->method);
    g_free(pt->params_json);
    g_free(pt);
    return;
  }
  pthread_detach(thread);
}

static void on_ping(const char *id, const char *req, void *arg)
{
  (void)req;
  _ipc_passthrough(arg, id, "system.ping", "{}");
}

static void on_catalog_query(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 2)
  {
    _return_error(ctx, id, "catalogQuery requires (offset, limit[, rules])");
    if(parser) g_object_unref(parser);
    return;
  }

  gint64 offset = json_array_get_int_element(args, 0);
  gint64 limit = json_array_get_int_element(args, 1);

  // Optional 3rd arg: rules array
  char *rules_json = NULL;
  if(json_array_get_length(args) >= 3)
  {
    JsonNode *rules_node = json_array_get_element(args, 2);
    if(rules_node && JSON_NODE_HOLDS_ARRAY(rules_node))
    {
      JsonGenerator *gen = json_generator_new();
      json_generator_set_root(gen, rules_node);
      rules_json = json_generator_to_data(gen, NULL);
      g_object_unref(gen);
    }
  }

  // Optional 4th arg: sort field, 5th arg: sort order
  const char *sort_field = NULL;
  const char *sort_order = NULL;
  if(json_array_get_length(args) >= 4)
  {
    JsonNode *n = json_array_get_element(args, 3);
    if(n && JSON_NODE_HOLDS_VALUE(n))
      sort_field = json_node_get_string(n);
  }
  if(json_array_get_length(args) >= 5)
  {
    JsonNode *n = json_array_get_element(args, 4);
    if(n && JSON_NODE_HOLDS_VALUE(n))
      sort_order = json_node_get_string(n);
  }
  g_object_unref(parser);

  GString *params_str = g_string_new("{");
  g_string_append_printf(params_str, "\"offset\":%" G_GINT64_FORMAT
                                     ",\"limit\":%" G_GINT64_FORMAT, offset, limit);
  if(rules_json)
  {
    g_string_append_printf(params_str, ",\"rules\":%s", rules_json);
    g_free(rules_json);
  }
  if(sort_field)
    g_string_append_printf(params_str, ",\"sort\":\"%s\"", sort_field);
  if(sort_order)
    g_string_append_printf(params_str, ",\"sort_order\":\"%s\"", sort_order);
  g_string_append_c(params_str, '}');

  char *params = g_string_free(params_str, FALSE);
  _ipc_passthrough(ctx, id, "catalog.query", params);
  g_free(params);
}

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
  if(pthread_create(&thread, NULL, _develop_open_worker, ar) != 0)
  {
    _return_error(ar->ctx, id, "internal error: failed to create worker thread");
    _async_req_free(ar);
    return;
  }
  pthread_detach(thread);
}

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
  if(pthread_create(&thread, NULL, _develop_close_worker, ar) != 0)
  {
    _return_error(ar->ctx, id, "internal error: failed to create worker thread");
    _async_req_free(ar);
    return;
  }
  pthread_detach(thread);
}

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

/* getPreviewFrame: reads pixels directly from SHM, no IPC */

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
  uint32_t stride = header->stride;
  size_t pixel_size = (size_t)stride * (size_t)h;
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
  if(pthread_create(&thread, NULL, _get_preview_frame_worker, ar) != 0)
  {
    _return_error(ar->ctx, id, "internal error: failed to create worker thread");
    _async_req_free(ar);
    return;
  }
  pthread_detach(thread);
}

/* ── Filesystem browsing (cross-platform via GLib) ───────────── */

/* Known RAW / image extensions (case-insensitive match) */
static const char *_raw_extensions[] = {
  "3fr", "ari", "arw", "bay", "braw", "cap", "cr2", "cr3",
  "crw", "dcr", "dcs", "dng", "drf", "eip", "erf", "fff",
  "gpr", "iiq", "k25", "kdc", "mdc", "mef", "mos", "mrw",
  "nef", "nrw", "obm", "orf", "pef", "ptx", "pxn", "r3d",
  "raf", "raw", "rw2", "rwl", "rwz", "sr2", "srf", "srw",
  "x3f",
  NULL
};

static const char *_image_extensions[] = {
  /* RAW formats are also valid images */
  "3fr", "ari", "arw", "bay", "braw", "cap", "cr2", "cr3",
  "crw", "dcr", "dcs", "dng", "drf", "eip", "erf", "fff",
  "gpr", "iiq", "k25", "kdc", "mdc", "mef", "mos", "mrw",
  "nef", "nrw", "obm", "orf", "pef", "ptx", "pxn", "r3d",
  "raf", "raw", "rw2", "rwl", "rwz", "sr2", "srf", "srw",
  "x3f",
  /* Common raster formats */
  "jpg", "jpeg", "tif", "tiff", "png", "exr", "hdr",
  "pfm", "pnm", "pbm", "pgm", "ppm",
  "j2k", "j2c", "jp2", "jpc",
  "avif", "heif", "heic", "hif",
  "webp",
  NULL
};

static int _is_extension_in(const char *filename, const char **ext_list)
{
  const char *dot = strrchr(filename, '.');
  if(!dot || dot == filename) return 0;
  const char *ext = dot + 1;
  for(const char **e = ext_list; *e; e++)
    if(g_ascii_strcasecmp(ext, *e) == 0) return 1;
  return 0;
}

static int _dir_has_subdirs(const char *path)
{
  GDir *d = g_dir_open(path, 0, NULL);
  if(!d) return 0;
  const gchar *name;
  while((name = g_dir_read_name(d)) != NULL)
  {
    if(name[0] == '.') continue;  // skip hidden
    gchar *child = g_build_filename(path, name, NULL);
    gboolean is_dir = g_file_test(child, G_FILE_TEST_IS_DIR);
    g_free(child);
    if(is_dir)
    {
      g_dir_close(d);
      return 1;
    }
  }
  g_dir_close(d);
  return 0;
}

/* listFolders(path) → [{name, path, hasChildren}, …] */
static void *_list_folders_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, ar->id, "listFolders requires (path)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  const char *path = json_array_get_string_element(args, 0);
  GError *err = NULL;
  GDir *d = g_dir_open(path, 0, &err);
  if(!d)
  {
    _return_error(ctx, ar->id, err ? err->message : "cannot open directory");
    g_clear_error(&err);
    g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  JsonBuilder *builder = json_builder_new();
  json_builder_begin_array(builder);

  const gchar *name;
  while((name = g_dir_read_name(d)) != NULL)
  {
    if(name[0] == '.') continue;  // skip hidden

    gchar *fullpath = g_build_filename(path, name, NULL);

    if(!g_file_test(fullpath, G_FILE_TEST_IS_DIR))
    {
      g_free(fullpath);
      continue;
    }

    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "name");
    json_builder_add_string_value(builder, name);
    json_builder_set_member_name(builder, "path");
    json_builder_add_string_value(builder, fullpath);
    json_builder_set_member_name(builder, "hasChildren");
    json_builder_add_boolean_value(builder, _dir_has_subdirs(fullpath));
    json_builder_end_object(builder);

    g_free(fullpath);
  }
  g_dir_close(d);

  json_builder_end_array(builder);
  JsonNode *root = json_builder_get_root(builder);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, root);
  char *json = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(root);
  g_object_unref(builder);
  g_object_unref(parser);

  _return_ok(ctx, ar->id, json);
  g_free(json);
  _async_req_free(ar);
  return NULL;
}

static void on_list_folders(const char *id, const char *req, void *arg)
{
  async_req_t *ar = _async_req_new(arg, id, req);
  pthread_t thread;
  if(pthread_create(&thread, NULL, _list_folders_worker, ar) != 0)
  {
    _return_error(ar->ctx, id, "internal error: failed to create worker thread");
    _async_req_free(ar);
    return;
  }
  pthread_detach(thread);
}

/* Helper to add files from a single directory to the builder array */
static void _collect_files(JsonBuilder *builder, const char *dir_path,
                           int recursive, int ignore_non_raw)
{
  GDir *d = g_dir_open(dir_path, 0, NULL);
  if(!d) return;

  const char **ext_list = ignore_non_raw ? _raw_extensions : _image_extensions;

  const gchar *name;
  while((name = g_dir_read_name(d)) != NULL)
  {
    if(name[0] == '.') continue;  // skip hidden

    gchar *fullpath = g_build_filename(dir_path, name, NULL);

    if(g_file_test(fullpath, G_FILE_TEST_IS_DIR))
    {
      if(recursive)
        _collect_files(builder, fullpath, recursive, ignore_non_raw);
      g_free(fullpath);
      continue;
    }

    /* Check file extension */
    if(!_is_extension_in(name, ext_list))
    {
      g_free(fullpath);
      continue;
    }

    GStatBuf st;
    if(g_stat(fullpath, &st) != 0)
    {
      g_free(fullpath);
      continue;
    }

    json_builder_begin_object(builder);
    json_builder_set_member_name(builder, "filename");
    json_builder_add_string_value(builder, name);
    json_builder_set_member_name(builder, "fullpath");
    json_builder_add_string_value(builder, fullpath);
    json_builder_set_member_name(builder, "modified");
    json_builder_add_int_value(builder, (gint64)st.st_mtime);
    json_builder_set_member_name(builder, "alreadyImported");
    json_builder_add_boolean_value(builder, FALSE);
    json_builder_end_object(builder);

    g_free(fullpath);
  }
  g_dir_close(d);
}

/** Query the server for which paths are already imported.
 *  Returns a GHashTable (set) of imported fullpath strings.
 *  Caller must g_hash_table_unref() the result. */
static GHashTable *_check_imported_paths(dt_webview_ctx_t *ctx, JsonArray *files_array)
{
  GHashTable *set = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);

  const guint len = json_array_get_length(files_array);
  if(len == 0 || ctx->socket_fd < 0) return set;

  /* Build params: {"paths":["...", ...]} */
  JsonBuilder *pb = json_builder_new();
  json_builder_begin_object(pb);
  json_builder_set_member_name(pb, "paths");
  json_builder_begin_array(pb);
  for(guint i = 0; i < len; i++)
  {
    JsonObject *file = json_array_get_object_element(files_array, i);
    const char *fp = json_object_get_string_member(file, "fullpath");
    if(fp) json_builder_add_string_value(pb, fp);
  }
  json_builder_end_array(pb);
  json_builder_end_object(pb);

  JsonNode *pnode = json_builder_get_root(pb);
  JsonGenerator *pgen = json_generator_new();
  json_generator_set_root(pgen, pnode);
  char *params_json = json_generator_to_data(pgen, NULL);
  g_object_unref(pgen);
  json_node_unref(pnode);
  g_object_unref(pb);

  char *error = NULL;
  char *result = dt_ipc_request(ctx->socket_fd, &ctx->ipc_mutex,
                                "catalog.check_imported", params_json, &error);
  g_free(params_json);

  if(result)
  {
    /* Parse {"imported":["path1","path2",...]} */
    JsonParser *rp = json_parser_new();
    if(json_parser_load_from_data(rp, result, -1, NULL))
    {
      JsonObject *robj = json_node_get_object(json_parser_get_root(rp));
      if(robj && json_object_has_member(robj, "imported"))
      {
        JsonArray *imported = json_object_get_array_member(robj, "imported");
        for(guint i = 0; i < json_array_get_length(imported); i++)
        {
          const char *p = json_array_get_string_element(imported, i);
          if(p) g_hash_table_add(set, g_strdup(p));
        }
      }
    }
    g_object_unref(rp);
    g_free(result);
  }
  else
  {
    fprintf(stderr, "[webview] check_imported failed: %s\n", error ? error : "unknown");
    g_free(error);
  }

  return set;
}

/* listFiles(path, recursive, ignoreNonRaw) → [{filename, fullpath, modified, alreadyImported}, …] */
static void *_list_files_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 3)
  {
    _return_error(ctx, ar->id, "listFiles requires (path, recursive, ignoreNonRaw)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  const char *path = json_array_get_string_element(args, 0);
  gboolean recursive = json_array_get_boolean_element(args, 1);
  gboolean ignore_non_raw = json_array_get_boolean_element(args, 2);

  /* Phase 1: collect files from the filesystem */
  JsonBuilder *builder = json_builder_new();
  json_builder_begin_array(builder);
  _collect_files(builder, path, recursive, ignore_non_raw);
  json_builder_end_array(builder);

  JsonNode *root = json_builder_get_root(builder);
  g_object_unref(builder);

  /* Phase 2: check which files are already imported via IPC */
  JsonArray *files_array = json_node_get_array(root);
  GHashTable *imported_set = _check_imported_paths(ctx, files_array);

  /* Phase 3: patch alreadyImported flags and serialize */
  for(guint i = 0; i < json_array_get_length(files_array); i++)
  {
    JsonObject *file = json_array_get_object_element(files_array, i);
    const char *fp = json_object_get_string_member(file, "fullpath");
    gboolean is_imported = fp && g_hash_table_contains(imported_set, fp);
    json_object_set_boolean_member(file, "alreadyImported", is_imported);
  }
  g_hash_table_unref(imported_set);

  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, root);
  char *json = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(root);
  g_object_unref(parser);

  _return_ok(ctx, ar->id, json);
  g_free(json);
  _async_req_free(ar);
  return NULL;
}

static void on_list_files(const char *id, const char *req, void *arg)
{
  async_req_t *ar = _async_req_new(arg, id, req);
  pthread_t thread;
  if(pthread_create(&thread, NULL, _list_files_worker, ar) != 0)
  {
    _return_error(ar->ctx, id, "internal error: failed to create worker thread");
    _async_req_free(ar);
    return;
  }
  pthread_detach(thread);
}

/* ── Environment helpers ─────────────────────────────────────── */

static void on_get_home_path(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;
  const char *home = g_get_home_dir();
  if(home)
  {
    JsonNode *node = json_node_new(JSON_NODE_VALUE);
    json_node_set_string(node, home);
    JsonGenerator *gen = json_generator_new();
    json_generator_set_root(gen, node);
    char *json = json_generator_to_data(gen, NULL);
    g_object_unref(gen);
    json_node_unref(node);
    _return_ok(ctx, id, json);
    g_free(json);
  }
  else
  {
    _return_ok(ctx, id, "\"/\"");
  }
}

/* ── Collection values via IPC ─────────────────────────────────── */

static void on_catalog_get_collection_values(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "catalogGetCollectionValues requires (property[, filter])");
    if(parser) g_object_unref(parser);
    return;
  }

  const char *property = json_array_get_string_element(args, 0);
  const char *filter = "";
  if(json_array_get_length(args) >= 2)
    filter = json_array_get_string_element(args, 1);

  /* Escape strings for JSON */
  JsonNode *pnode = json_node_new(JSON_NODE_VALUE);
  json_node_set_string(pnode, property);
  JsonNode *fnode = json_node_new(JSON_NODE_VALUE);
  json_node_set_string(fnode, filter);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, pnode);
  char *prop_esc = json_generator_to_data(gen, NULL);
  json_generator_set_root(gen, fnode);
  char *filt_esc = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(pnode);
  json_node_unref(fnode);

  char *params = g_strdup_printf("{\"property\":%s,\"filter\":%s}", prop_esc, filt_esc);
  g_free(prop_esc);
  g_free(filt_esc);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, "catalog.get_collection_values", params);
  g_free(params);
}

/* Existing filmrolls/tags endpoints are already served via server routes.
   We add simple passthrough bindings for them. */

static void on_catalog_get_filmrolls(const char *id, const char *req, void *arg)
{
  (void)req;
  _ipc_passthrough(arg, id, "catalog.get_filmrolls", "{}");
}

static void on_catalog_get_tags(const char *id, const char *req, void *arg)
{
  (void)req;
  _ipc_passthrough(arg, id, "catalog.get_tags", "{}");
}

/* ── Platform info ─────────────────────────────────────────────── */

static void on_get_platform_info(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;
#if defined(__APPLE__)
  _return_ok(ctx, id, "{\"os\":\"macos\"}");
#elif defined(_WIN32)
  _return_ok(ctx, id, "{\"os\":\"windows\"}");
#else
  _return_ok(ctx, id, "{\"os\":\"linux\"}");
#endif
}

/* ── Import images via IPC ────────────────────────────────────── */

static void *_import_images_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, ar->id, "importImages requires (paths)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  /* args[0] is an array of fullpath strings */
  JsonArray *paths = json_array_get_array_element(args, 0);

  /* Build params: {"paths":["...", ...]} */
  JsonBuilder *pb = json_builder_new();
  json_builder_begin_object(pb);
  json_builder_set_member_name(pb, "paths");
  json_builder_begin_array(pb);
  for(guint i = 0; i < json_array_get_length(paths); i++)
  {
    const char *p = json_array_get_string_element(paths, i);
    if(p) json_builder_add_string_value(pb, p);
  }
  json_builder_end_array(pb);
  json_builder_end_object(pb);

  JsonNode *pnode = json_builder_get_root(pb);
  JsonGenerator *pgen = json_generator_new();
  json_generator_set_root(pgen, pnode);
  char *params_json = json_generator_to_data(pgen, NULL);
  g_object_unref(pgen);
  json_node_unref(pnode);
  g_object_unref(pb);

  char *error = NULL;
  char *result = dt_ipc_request(ctx->socket_fd, &ctx->ipc_mutex,
                                "catalog.import", params_json, &error);
  g_free(params_json);

  if(result)
  {
    _return_ok(ctx, ar->id, result);
    g_free(result);
  }
  else
  {
    _return_error(ctx, ar->id, error ? error : "import failed");
    g_free(error);
  }

  g_object_unref(parser);
  _async_req_free(ar);
  return NULL;
}

static void on_import_images(const char *id, const char *req, void *arg)
{
  async_req_t *ar = _async_req_new(arg, id, req);
  pthread_t thread;
  if(pthread_create(&thread, NULL, _import_images_worker, ar) != 0)
  {
    _return_error(ar->ctx, id, "internal error: failed to create worker thread");
    _async_req_free(ar);
    return;
  }
  pthread_detach(thread);
}

/* ── Copy & import images via IPC ─────────────────────────────── */

static void *_copy_import_images_worker(void *arg)
{
  async_req_t *ar = arg;
  dt_webview_ctx_t *ctx = ar->ctx;

  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(ar->req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, ar->id, "copyAndImportImages requires (paths)");
    if(parser) g_object_unref(parser);
    _async_req_free(ar);
    return NULL;
  }

  /* args[0] is an array of fullpath strings */
  JsonArray *paths = json_array_get_array_element(args, 0);

  /* Build params: {"paths":["...", ...]} */
  JsonBuilder *pb = json_builder_new();
  json_builder_begin_object(pb);
  json_builder_set_member_name(pb, "paths");
  json_builder_begin_array(pb);
  for(guint i = 0; i < json_array_get_length(paths); i++)
  {
    const char *p = json_array_get_string_element(paths, i);
    if(p) json_builder_add_string_value(pb, p);
  }
  json_builder_end_array(pb);
  json_builder_end_object(pb);

  JsonNode *pnode = json_builder_get_root(pb);
  JsonGenerator *pgen = json_generator_new();
  json_generator_set_root(pgen, pnode);
  char *params_json = json_generator_to_data(pgen, NULL);
  g_object_unref(pgen);
  json_node_unref(pnode);
  g_object_unref(pb);

  char *error = NULL;
  char *result = dt_ipc_request(ctx->socket_fd, &ctx->ipc_mutex,
                                "catalog.copy_import", params_json, &error);
  g_free(params_json);

  if(result)
  {
    _return_ok(ctx, ar->id, result);
    g_free(result);
  }
  else
  {
    _return_error(ctx, ar->id, error ? error : "copy_import failed");
    g_free(error);
  }

  g_object_unref(parser);
  _async_req_free(ar);
  return NULL;
}

static void on_copy_import_images(const char *id, const char *req, void *arg)
{
  async_req_t *ar = _async_req_new(arg, id, req);
  pthread_t thread;
  if(pthread_create(&thread, NULL, _copy_import_images_worker, ar) != 0)
  {
    _return_error(ar->ctx, id, "internal error: failed to create worker thread");
    _async_req_free(ar);
    return;
  }
  pthread_detach(thread);
}

/* ── File thumbnail (embedded EXIF preview) via IPC ───────────── */

static void on_get_file_thumbnail(const char *id, const char *req, void *arg)
{
  dt_webview_ctx_t *ctx = arg;
  JsonParser *parser = NULL;
  JsonArray *args = _parse_args(req, &parser);
  if(!args || json_array_get_length(args) < 1)
  {
    _return_error(ctx, id, "getFileThumbnail requires (path)");
    if(parser) g_object_unref(parser);
    return;
  }

  const char *path = json_array_get_string_element(args, 0);

  /* Escape path for JSON */
  JsonNode *node = json_node_new(JSON_NODE_VALUE);
  json_node_set_string(node, path);
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, node);
  char *escaped = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  json_node_unref(node);

  char *params = g_strdup_printf("{\"path\":%s}", escaped);
  g_free(escaped);
  g_object_unref(parser);

  _ipc_passthrough(ctx, id, "catalog.get_file_thumbnail", params);
  g_free(params);
}

/* ── Window titlebar actions ──────────────────────────────────── */

static void on_window_start_drag(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;
  dt_titlebar_start_drag(ctx->webview);
  _return_ok(ctx, id, "null");
}

static void on_window_zoom(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;
  dt_titlebar_zoom(ctx->webview);
  _return_ok(ctx, id, "null");
}

/* ── Native folder picker (via nativefiledialog-extended) ─────── */

static void on_pick_folder(const char *id, const char *req, void *arg)
{
  (void)req;
  dt_webview_ctx_t *ctx = arg;

  nfdchar_t *out_path = NULL;
  nfdresult_t result = NFD_PickFolder(&out_path, NULL);

  if(result == NFD_OKAY)
  {
    /* Return the path as a JSON string */
    JsonNode *node = json_node_new(JSON_NODE_VALUE);
    json_node_set_string(node, out_path);
    JsonGenerator *gen = json_generator_new();
    json_generator_set_root(gen, node);
    char *json = json_generator_to_data(gen, NULL);
    g_object_unref(gen);
    json_node_unref(node);

    _return_ok(ctx, id, json);
    g_free(json);
    NFD_FreePath(out_path);
  }
  else if(result == NFD_CANCEL)
  {
    _return_ok(ctx, id, "null");
  }
  else
  {
    _return_error(ctx, id, NFD_GetError());
  }
}

void dt_webview_register_bindings(dt_webview_ctx_t *ctx)
{
  /* Initialize NFD once */
  NFD_Init();

  webview_bind(ctx->webview, "ping", on_ping, ctx);
  webview_bind(ctx->webview, "catalogQuery", on_catalog_query, ctx);
  webview_bind(ctx->webview, "catalogGetThumbnail", on_catalog_get_thumbnail, ctx);
  webview_bind(ctx->webview, "developOpen", on_develop_open, ctx);
  webview_bind(ctx->webview, "developClose", on_develop_close, ctx);
  webview_bind(ctx->webview, "developSetParams", on_develop_set_params, ctx);
  webview_bind(ctx->webview, "developRequestPreview", on_develop_request_preview, ctx);
  webview_bind(ctx->webview, "getPreviewFrame", on_get_preview_frame, ctx);
  webview_bind(ctx->webview, "pickFolder", on_pick_folder, ctx);
  webview_bind(ctx->webview, "listFolders", on_list_folders, ctx);
  webview_bind(ctx->webview, "listFiles", on_list_files, ctx);
  webview_bind(ctx->webview, "getHomePath", on_get_home_path, ctx);
  webview_bind(ctx->webview, "importImages", on_import_images, ctx);
  webview_bind(ctx->webview, "copyAndImportImages", on_copy_import_images, ctx);
  webview_bind(ctx->webview, "getFileThumbnail", on_get_file_thumbnail, ctx);
  webview_bind(ctx->webview, "getPlatformInfo", on_get_platform_info, ctx);
  webview_bind(ctx->webview, "catalogGetCollectionValues", on_catalog_get_collection_values, ctx);
  webview_bind(ctx->webview, "catalogGetFilmrolls", on_catalog_get_filmrolls, ctx);
  webview_bind(ctx->webview, "catalogGetTags", on_catalog_get_tags, ctx);
}

void dt_webview_register_window_bindings(dt_webview_ctx_t *ctx)
{
  webview_bind(ctx->webview, "windowStartDrag", on_window_start_drag, ctx);
  webview_bind(ctx->webview, "windowZoom", on_window_zoom, ctx);
}
