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
#include <stdint.h>
#include <sys/types.h>

// Maximum message size: 16 MB (thumbnails can be large as base64)
#define DT_SERVER_MAX_MESSAGE_SIZE (16 * 1024 * 1024)

// Default preview resolution cap — covers 95% of displays and saves ~50 MB vs 4K
#define DT_SERVER_MAX_PREVIEW_WIDTH  1920
#define DT_SERVER_MAX_PREVIEW_HEIGHT 1080

#define DT_SHM_MAGIC 0x44545348   // "DTSH"
#define DT_SHM_VERSION 1
#define DT_SHM_HEADER_SIZE 64

typedef enum dt_shm_format_t
{
  DT_SHM_FORMAT_BGRA8 = 0,
  DT_SHM_FORMAT_RGBAF32 = 1,
} dt_shm_format_t;

typedef struct dt_shm_header_t
{
  uint32_t magic;
  uint32_t version;
  uint32_t width;
  uint32_t height;
  uint32_t stride;       // bytes per row
  uint32_t format;       // dt_shm_format_t
  uint64_t sequence;     // monotonic frame counter
  uint32_t ready;        // atomic: 0 = writing, 1 = readable
  uint32_t reserved[7];  // pad to 64 bytes
  // pixel data follows at offset DT_SHM_HEADER_SIZE
} dt_shm_header_t;

typedef struct dt_shm_buffer_t
{
  char name[64];           // e.g. "/dt-prev-dev001-0"
  int fd;
  size_t size;             // total mapped size (header + pixels)
  dt_shm_header_t *mapped; // mmap pointer
} dt_shm_buffer_t;

// SHM operations
gboolean dt_shm_create(dt_shm_buffer_t *buf, const char *name, uint32_t width, uint32_t height);
void dt_shm_destroy(dt_shm_buffer_t *buf);
void dt_shm_write_header(dt_shm_buffer_t *buf, uint32_t w, uint32_t h,
                          dt_shm_format_t fmt, uint64_t seq);
uint8_t *dt_shm_pixel_data(dt_shm_buffer_t *buf);

// error codes (JSON-RPC style)
#define DT_SERVER_ERR_PARSE       -32700
#define DT_SERVER_ERR_METHOD      -32601
#define DT_SERVER_ERR_PARAMS      -32602
#define DT_SERVER_ERR_INTERNAL    -32603
#define DT_SERVER_ERR_NOT_FOUND   -1
#define DT_SERVER_ERR_BUSY        -2
#define DT_SERVER_ERR_AUTH        -3

typedef struct dt_server_request_t
{
  char *id;            // request ID for correlation
  char *method;        // e.g. "catalog.query"
  JsonObject *params;  // method parameters (borrowed from parsed node)
  JsonNode *root;      // owns the parsed tree
} dt_server_request_t;

// Frame I/O: 4-byte big-endian length prefix + JSON payload
gboolean dt_server_read_frame(int fd, char **out_buf, size_t *out_len);
gboolean dt_server_write_frame(int fd, const char *buf, size_t len);

// Parse incoming request
dt_server_request_t *dt_server_parse_request(const char *json, size_t len);
void dt_server_free_request(dt_server_request_t *req);

// Build response JSON strings (caller must g_free)
char *dt_server_make_response(const char *id, JsonNode *result);
char *dt_server_make_error(const char *id, int code, const char *message);
char *dt_server_make_event(const char *event_name, JsonNode *data);
