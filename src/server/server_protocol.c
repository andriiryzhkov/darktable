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

#include "server/server_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

gboolean dt_shm_create(dt_shm_buffer_t *buf, const char *name, uint32_t width, uint32_t height)
{
  memset(buf, 0, sizeof(*buf));
  g_strlcpy(buf->name, name, sizeof(buf->name));

  const size_t stride = (size_t)width * 4;
  buf->size = DT_SHM_HEADER_SIZE + stride * height;

  buf->fd = shm_open(name, O_CREAT | O_RDWR, 0600);
  if(buf->fd < 0)
  {
    fprintf(stderr, "[server] shm_open(%s) failed: %s\n", name, strerror(errno));
    return FALSE;
  }

  if(ftruncate(buf->fd, buf->size) < 0)
  {
    fprintf(stderr, "[server] ftruncate(%s, %zu) failed: %s\n", name, buf->size, strerror(errno));
    close(buf->fd);
    shm_unlink(name);
    buf->fd = -1;
    return FALSE;
  }

  buf->mapped = mmap(NULL, buf->size, PROT_READ | PROT_WRITE, MAP_SHARED, buf->fd, 0);
  if(buf->mapped == MAP_FAILED)
  {
    fprintf(stderr, "[server] mmap(%s) failed: %s\n", name, strerror(errno));
    close(buf->fd);
    shm_unlink(name);
    buf->mapped = NULL;
    buf->fd = -1;
    return FALSE;
  }

  // Initialize header
  memset(buf->mapped, 0, DT_SHM_HEADER_SIZE);
  buf->mapped->magic = DT_SHM_MAGIC;
  buf->mapped->version = DT_SHM_VERSION;
  buf->mapped->width = width;
  buf->mapped->height = height;
  buf->mapped->stride = (uint32_t)stride;
  buf->mapped->format = DT_SHM_FORMAT_BGRA8;

  return TRUE;
}

void dt_shm_destroy(dt_shm_buffer_t *buf)
{
  if(buf->mapped && buf->mapped != MAP_FAILED)
  {
    munmap(buf->mapped, buf->size);
    buf->mapped = NULL;
  }
  if(buf->fd >= 0)
  {
    close(buf->fd);
    buf->fd = -1;
  }
  if(buf->name[0])
  {
    shm_unlink(buf->name);
    buf->name[0] = '\0';
  }
}

void dt_shm_write_header(dt_shm_buffer_t *buf, uint32_t w, uint32_t h,
                          dt_shm_format_t fmt, uint64_t seq)
{
  // Mark not ready while writing
  __atomic_store_n(&buf->mapped->ready, 0, __ATOMIC_RELEASE);

  buf->mapped->width = w;
  buf->mapped->height = h;
  buf->mapped->stride = w * 4; // BGRA8 = 4 bytes per pixel
  buf->mapped->format = (uint32_t)fmt;
  buf->mapped->sequence = seq;
}

uint8_t *dt_shm_pixel_data(dt_shm_buffer_t *buf)
{
  return ((uint8_t *)buf->mapped) + DT_SHM_HEADER_SIZE;
}

// read exactly n bytes from fd, handling partial reads
static gboolean _read_exact(int fd, void *buf, size_t n)
{
  size_t total = 0;
  while(total < n)
  {
    ssize_t r = read(fd, (char *)buf + total, n - total);
    if(r <= 0)
    {
      if(r < 0 && errno == EINTR) continue;
      return FALSE;
    }
    total += r;
  }
  return TRUE;
}

// write exactly n bytes to fd, handling partial writes
static gboolean _write_exact(int fd, const void *buf, size_t n)
{
  size_t total = 0;
  while(total < n)
  {
    ssize_t w = write(fd, (const char *)buf + total, n - total);
    if(w <= 0)
    {
      if(w < 0 && errno == EINTR) continue;
      return FALSE;
    }
    total += w;
  }
  return TRUE;
}

gboolean dt_server_read_frame(int fd, char **out_buf, size_t *out_len)
{
  uint8_t len_buf[4];
  if(!_read_exact(fd, len_buf, 4))
    return FALSE;

  uint32_t len = ((uint32_t)len_buf[0] << 24)
               | ((uint32_t)len_buf[1] << 16)
               | ((uint32_t)len_buf[2] << 8)
               | ((uint32_t)len_buf[3]);

  if(len == 0 || len > DT_SERVER_MAX_MESSAGE_SIZE)
  {
    fprintf(stderr, "[server] invalid frame length: %u\n", len);
    return FALSE;
  }

  char *buf = g_malloc(len + 1);
  if(!_read_exact(fd, buf, len))
  {
    g_free(buf);
    return FALSE;
  }
  buf[len] = '\0';

  *out_buf = buf;
  *out_len = len;
  return TRUE;
}

gboolean dt_server_write_frame(int fd, const char *buf, size_t len)
{
  uint32_t net_len = (uint32_t)len;
  uint8_t len_buf[4];
  len_buf[0] = (net_len >> 24) & 0xFF;
  len_buf[1] = (net_len >> 16) & 0xFF;
  len_buf[2] = (net_len >> 8) & 0xFF;
  len_buf[3] = net_len & 0xFF;

  if(!_write_exact(fd, len_buf, 4))
    return FALSE;
  if(!_write_exact(fd, buf, len))
    return FALSE;
  return TRUE;
}

dt_server_request_t *dt_server_parse_request(const char *json, size_t len)
{
  JsonParser *parser = json_parser_new();
  GError *error = NULL;

  if(!json_parser_load_from_data(parser, json, len, &error))
  {
    fprintf(stderr, "[server] JSON parse error: %s\n", error->message);
    g_error_free(error);
    g_object_unref(parser);
    return NULL;
  }

  JsonNode *root = json_parser_get_root(parser);
  if(!root || !JSON_NODE_HOLDS_OBJECT(root))
  {
    g_object_unref(parser);
    return NULL;
  }

  JsonObject *obj = json_node_get_object(root);

  const char *id = NULL;
  if(json_object_has_member(obj, "id"))
    id = json_object_get_string_member(obj, "id");

  const char *method = NULL;
  if(json_object_has_member(obj, "method"))
    method = json_object_get_string_member(obj, "method");

  if(!method)
  {
    g_object_unref(parser);
    return NULL;
  }

  JsonObject *params = NULL;
  if(json_object_has_member(obj, "params"))
  {
    JsonNode *params_node = json_object_get_member(obj, "params");
    if(JSON_NODE_HOLDS_OBJECT(params_node))
      params = json_node_get_object(params_node);
  }

  dt_server_request_t *req = g_new0(dt_server_request_t, 1);
  req->id = g_strdup(id ? id : "");
  req->method = g_strdup(method);
  req->params = params;  // borrowed — valid while root is alive
  // Transfer ownership of parsed tree: steal root and keep parser alive via root
  req->root = json_node_ref(root);
  g_object_unref(parser);

  return req;
}

void dt_server_free_request(dt_server_request_t *req)
{
  if(!req) return;
  g_free(req->id);
  g_free(req->method);
  if(req->root) json_node_unref(req->root);
  g_free(req);
}

// serialize a JsonNode to a compact JSON string
static char *_node_to_json(JsonNode *node)
{
  JsonGenerator *gen = json_generator_new();
  json_generator_set_root(gen, node);
  char *str = json_generator_to_data(gen, NULL);
  g_object_unref(gen);
  return str;
}

char *dt_server_make_response(const char *id, JsonNode *result)
{
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "id");
  json_builder_add_string_value(b, id ? id : "");
  json_builder_set_member_name(b, "result");
  if(result)
    json_builder_add_value(b, result);
  else
    json_builder_add_null_value(b);
  json_builder_set_member_name(b, "error");
  json_builder_add_null_value(b);
  json_builder_end_object(b);

  JsonNode *root = json_builder_get_root(b);
  char *json = _node_to_json(root);
  json_node_unref(root);
  g_object_unref(b);
  return json;
}

char *dt_server_make_error(const char *id, int code, const char *message)
{
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "id");
  json_builder_add_string_value(b, id ? id : "");
  json_builder_set_member_name(b, "result");
  json_builder_add_null_value(b);
  json_builder_set_member_name(b, "error");
  json_builder_begin_object(b);
    json_builder_set_member_name(b, "code");
    json_builder_add_int_value(b, code);
    json_builder_set_member_name(b, "message");
    json_builder_add_string_value(b, message ? message : "Unknown error");
  json_builder_end_object(b);
  json_builder_end_object(b);

  JsonNode *root = json_builder_get_root(b);
  char *json = _node_to_json(root);
  json_node_unref(root);
  g_object_unref(b);
  return json;
}

char *dt_server_make_event(const char *event_name, JsonNode *data)
{
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "id");
  json_builder_add_null_value(b);
  json_builder_set_member_name(b, "event");
  json_builder_add_string_value(b, event_name);
  json_builder_set_member_name(b, "data");
  if(data)
    json_builder_add_value(b, data);
  else
    json_builder_add_null_value(b);
  json_builder_end_object(b);

  JsonNode *root = json_builder_get_root(b);
  char *json = _node_to_json(root);
  json_node_unref(root);
  g_object_unref(b);
  return json;
}
