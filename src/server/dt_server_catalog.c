/*
    This file is part of darktable,
    Copyright (C) 2025 darktable developers.

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

#include "server/dt_server.h"
#include "common/database.h"
#include "common/debug.h"
#include "common/image.h"
#include "common/image_cache.h"
#include "common/mipmap_cache.h"

#include <sqlite3.h>

char *dt_server_catalog_query(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  int offset = 0;
  int limit = 100;
  if(req->params)
  {
    if(json_object_has_member(req->params, "offset"))
      offset = (int)json_object_get_int_member(req->params, "offset");
    if(json_object_has_member(req->params, "limit"))
      limit = (int)json_object_get_int_member(req->params, "limit");
  }

  // Clamp limit
  if(limit < 1) limit = 1;
  if(limit > 1000) limit = 1000;

  sqlite3_stmt *stmt = NULL;
  // Query the images table directly
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "SELECT i.id, i.film_id, i.filename, i.datetime_taken,"
    "       i.flags, i.width, i.height, i.aspect_ratio,"
    "       i.exposure, i.aperture, i.iso, i.focal_length,"
    "       f.folder"
    " FROM main.images AS i"
    " LEFT JOIN main.film_rolls AS f ON i.film_id = f.id"
    " ORDER BY i.datetime_taken DESC"
    " LIMIT ?1 OFFSET ?2",
    -1, &stmt, NULL);
  // clang-format on

  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, limit);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, offset);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "images");
  json_builder_begin_array(b);

  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    json_builder_begin_object(b);

    json_builder_set_member_name(b, "id");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 0));

    json_builder_set_member_name(b, "film_id");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 1));

    json_builder_set_member_name(b, "filename");
    const char *fname = (const char *)sqlite3_column_text(stmt, 2);
    json_builder_add_string_value(b, fname ? fname : "");

    json_builder_set_member_name(b, "datetime_taken");
    json_builder_add_int_value(b, sqlite3_column_int64(stmt, 3));

    json_builder_set_member_name(b, "flags");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 4));

    json_builder_set_member_name(b, "width");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 5));

    json_builder_set_member_name(b, "height");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 6));

    json_builder_set_member_name(b, "aspect_ratio");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 7));

    json_builder_set_member_name(b, "exposure");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 8));

    json_builder_set_member_name(b, "aperture");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 9));

    json_builder_set_member_name(b, "iso");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 10));

    json_builder_set_member_name(b, "focal_length");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 11));

    json_builder_set_member_name(b, "folder");
    const char *folder = (const char *)sqlite3_column_text(stmt, 12);
    json_builder_add_string_value(b, folder ? folder : "");

    json_builder_end_object(b);
  }
  sqlite3_finalize(stmt);

  json_builder_end_array(b);

  // Also return total count
  sqlite3_stmt *count_stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "SELECT COUNT(*) FROM main.images",
    -1, &count_stmt, NULL);
  int total = 0;
  if(sqlite3_step(count_stmt) == SQLITE_ROW)
    total = sqlite3_column_int(count_stmt, 0);
  sqlite3_finalize(count_stmt);

  json_builder_set_member_name(b, "total");
  json_builder_add_int_value(b, total);

  json_builder_set_member_name(b, "offset");
  json_builder_add_int_value(b, offset);

  json_builder_set_member_name(b, "limit");
  json_builder_add_int_value(b, limit);

  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_catalog_get_image(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  if(!req->params || !json_object_has_member(req->params, "imgid"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgid parameter");

  const dt_imgid_t imgid = (dt_imgid_t)json_object_get_int_member(req->params, "imgid");

  const dt_image_t *img = dt_image_cache_get(imgid, 'r');
  if(!img)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Image not found");

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "id");
  json_builder_add_int_value(b, imgid);

  json_builder_set_member_name(b, "filename");
  json_builder_add_string_value(b, img->filename);

  json_builder_set_member_name(b, "width");
  json_builder_add_int_value(b, img->width);

  json_builder_set_member_name(b, "height");
  json_builder_add_int_value(b, img->height);

  json_builder_set_member_name(b, "exposure");
  json_builder_add_double_value(b, img->exif_exposure);

  json_builder_set_member_name(b, "aperture");
  json_builder_add_double_value(b, img->exif_aperture);

  json_builder_set_member_name(b, "iso");
  json_builder_add_double_value(b, img->exif_iso);

  json_builder_set_member_name(b, "focal_length");
  json_builder_add_double_value(b, img->exif_focal_length);

  json_builder_set_member_name(b, "maker");
  json_builder_add_string_value(b, img->exif_maker);

  json_builder_set_member_name(b, "model");
  json_builder_add_string_value(b, img->exif_model);

  json_builder_set_member_name(b, "lens");
  json_builder_add_string_value(b, img->exif_lens);

  json_builder_set_member_name(b, "aspect_ratio");
  json_builder_add_double_value(b, img->aspect_ratio);

  json_builder_end_object(b);

  dt_image_cache_read_release(img);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_catalog_get_thumbnail(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  // TODO: implement thumbnail retrieval via mipmap_cache + base64 JPEG encoding
  return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                               "catalog.get_thumbnail not yet implemented");
}
