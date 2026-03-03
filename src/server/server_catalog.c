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

#include "server/server.h"
#include "common/database.h"
#include "common/datetime.h"
#include "common/debug.h"
#include "common/exif.h"
#include "common/film.h"
#include "common/image.h"
#include "common/image_cache.h"
#include "common/import_session.h"
#include "common/metadata.h"
#include "common/mipmap_cache.h"
#include "control/conf.h"
#include "control/signal.h"
#include "imageio/imageio_jpeg.h"

#include <sqlite3.h>
#include <sys/stat.h>
#include <sys/time.h>

char *dt_server_catalog_query(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  int offset = 0;
  int limit = 100;
  int filter_film_id = -1;
  int filter_tag_id = -1;
  int filter_rating_min = -1;
  const char *filter_text = NULL;

  if(req->params)
  {
    if(json_object_has_member(req->params, "offset"))
      offset = (int)json_object_get_int_member(req->params, "offset");
    if(json_object_has_member(req->params, "limit"))
      limit = (int)json_object_get_int_member(req->params, "limit");
    if(json_object_has_member(req->params, "film_id"))
      filter_film_id = (int)json_object_get_int_member(req->params, "film_id");
    if(json_object_has_member(req->params, "tag_id"))
      filter_tag_id = (int)json_object_get_int_member(req->params, "tag_id");
    if(json_object_has_member(req->params, "rating_min"))
      filter_rating_min = (int)json_object_get_int_member(req->params, "rating_min");
    if(json_object_has_member(req->params, "text"))
      filter_text = json_object_get_string_member(req->params, "text");
  }

  // Clamp limit
  if(limit < 1) limit = 1;
  if(limit > 1000) limit = 1000;

  // Build query dynamically based on filters
  GString *query = g_string_new(
    "SELECT i.id, i.film_id, i.filename, i.datetime_taken,"
    "       i.flags, i.width, i.height, i.aspect_ratio,"
    "       i.exposure, i.aperture, i.iso, i.focal_length,"
    "       f.folder"
    " FROM main.images AS i"
    " LEFT JOIN main.film_rolls AS f ON i.film_id = f.id");

  if(filter_tag_id >= 0)
    g_string_append(query,
      " JOIN main.tagged_images AS ti ON i.id = ti.imgid");

  g_string_append(query, " WHERE 1=1");

  if(filter_film_id >= 0)
    g_string_append(query, " AND i.film_id = ?3");

  if(filter_tag_id >= 0)
    g_string_append(query, " AND ti.tagid = ?4");

  if(filter_rating_min >= 0)
    g_string_append_printf(query, " AND (i.flags & 7) >= ?5");

  if(filter_text && filter_text[0])
    g_string_append(query, " AND i.filename LIKE ?6");

  g_string_append(query, " ORDER BY i.datetime_taken DESC LIMIT ?1 OFFSET ?2");

  // Build matching count query
  GString *count_query = g_string_new(
    "SELECT COUNT(*)"
    " FROM main.images AS i");

  if(filter_tag_id >= 0)
    g_string_append(count_query,
      " JOIN main.tagged_images AS ti ON i.id = ti.imgid");

  g_string_append(count_query, " WHERE 1=1");

  if(filter_film_id >= 0)
    g_string_append(count_query, " AND i.film_id = ?3");

  if(filter_tag_id >= 0)
    g_string_append(count_query, " AND ti.tagid = ?4");

  if(filter_rating_min >= 0)
    g_string_append(count_query, " AND (i.flags & 7) >= ?5");

  if(filter_text && filter_text[0])
    g_string_append(count_query, " AND i.filename LIKE ?6");

  // Prepare and bind the main query
  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    query->str, -1, &stmt, NULL);
  g_string_free(query, TRUE);

  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, limit);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, offset);

  // Prepare text filter pattern once (used for both queries)
  gchar *text_pattern = NULL;
  if(filter_text && filter_text[0])
    text_pattern = g_strdup_printf("%%%s%%", filter_text);

  if(filter_film_id >= 0)
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 3, filter_film_id);
  if(filter_tag_id >= 0)
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 4, filter_tag_id);
  if(filter_rating_min >= 0)
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 5, filter_rating_min);
  if(text_pattern)
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 6, text_pattern, -1, SQLITE_TRANSIENT);

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

  // Run the matching count query
  sqlite3_stmt *count_stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    count_query->str, -1, &count_stmt, NULL);
  g_string_free(count_query, TRUE);

  if(filter_film_id >= 0)
    DT_DEBUG_SQLITE3_BIND_INT(count_stmt, 3, filter_film_id);
  if(filter_tag_id >= 0)
    DT_DEBUG_SQLITE3_BIND_INT(count_stmt, 4, filter_tag_id);
  if(filter_rating_min >= 0)
    DT_DEBUG_SQLITE3_BIND_INT(count_stmt, 5, filter_rating_min);
  if(text_pattern)
    DT_DEBUG_SQLITE3_BIND_TEXT(count_stmt, 6, text_pattern, -1, SQLITE_TRANSIENT);

  int total = 0;
  if(sqlite3_step(count_stmt) == SQLITE_ROW)
    total = sqlite3_column_int(count_stmt, 0);
  sqlite3_finalize(count_stmt);

  g_free(text_pattern);

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

  if(!req->params || !json_object_has_member(req->params, "imgid"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgid parameter");

  const dt_imgid_t imgid = (dt_imgid_t)json_object_get_int_member(req->params, "imgid");

  // Determine requested thumbnail size (default 720px on longest edge)
  int requested_size = 720;
  if(json_object_has_member(req->params, "size"))
    requested_size = (int)json_object_get_int_member(req->params, "size");
  if(requested_size < 64) requested_size = 64;
  if(requested_size > 1920) requested_size = 1920;

  // Find the best matching mipmap level for the requested size
  const dt_mipmap_size_t mip = dt_mipmap_cache_get_matching_size(requested_size, requested_size);

  // Get the thumbnail from the mipmap cache (blocking)
  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(&buf, imgid, mip, DT_MIPMAP_BLOCKING, 'r');

  if(!buf.buf || buf.width <= 0 || buf.height <= 0)
  {
    dt_mipmap_cache_release(&buf);
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND,
                                 "Thumbnail not available");
  }

  // Compress RGBA buffer to JPEG
  // Output buffer: worst case is slightly larger than input for tiny images
  const size_t jpeg_max_size = (size_t)buf.width * buf.height * 4 + 1024;
  uint8_t *jpeg_buf = g_malloc(jpeg_max_size);

  const int quality = 85;
  const int jpeg_size = dt_imageio_jpeg_compress(buf.buf, jpeg_buf,
                                                  buf.width, buf.height, quality);
  const int thumb_width = buf.width;
  const int thumb_height = buf.height;

  dt_mipmap_cache_release(&buf);

  if(jpeg_size <= 0)
  {
    g_free(jpeg_buf);
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "JPEG compression failed");
  }

  // Base64-encode the JPEG data
  gchar *b64 = g_base64_encode(jpeg_buf, jpeg_size);
  g_free(jpeg_buf);

  // Build JSON response
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "imgid");
  json_builder_add_int_value(b, imgid);

  json_builder_set_member_name(b, "width");
  json_builder_add_int_value(b, thumb_width);

  json_builder_set_member_name(b, "height");
  json_builder_add_int_value(b, thumb_height);

  json_builder_set_member_name(b, "format");
  json_builder_add_string_value(b, "jpeg");

  json_builder_set_member_name(b, "encoding");
  json_builder_add_string_value(b, "base64");

  json_builder_set_member_name(b, "data");
  json_builder_add_string_value(b, b64);

  json_builder_end_object(b);
  g_free(b64);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_catalog_get_tags(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  // If imgid is provided, return tags for that image.
  // Otherwise, return all tags in the database.
  const gboolean has_imgid = req->params && json_object_has_member(req->params, "imgid");

  sqlite3_stmt *stmt = NULL;
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "tags");
  json_builder_begin_array(b);

  if(has_imgid)
  {
    const dt_imgid_t imgid = (dt_imgid_t)json_object_get_int_member(req->params, "imgid");

    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT T.id, T.name, T.flags"
      " FROM data.tags AS T"
      " JOIN main.tagged_images AS TI ON T.id = TI.tagid"
      " WHERE TI.imgid = ?1"
      " ORDER BY T.name",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  }
  else
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT T.id, T.name, T.flags"
      " FROM data.tags AS T"
      " ORDER BY T.name",
      -1, &stmt, NULL);
    // clang-format on
  }

  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    json_builder_begin_object(b);

    json_builder_set_member_name(b, "id");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 0));

    json_builder_set_member_name(b, "name");
    const char *name = (const char *)sqlite3_column_text(stmt, 1);
    json_builder_add_string_value(b, name ? name : "");

    json_builder_set_member_name(b, "flags");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 2));

    json_builder_end_object(b);
  }
  sqlite3_finalize(stmt);

  json_builder_end_array(b);
  json_builder_end_object(b);

  JsonNode *tags_result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, tags_result);
  json_node_unref(tags_result);
  g_object_unref(b);
  return resp;
}

char *dt_server_catalog_get_filmrolls(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  (void)req;

  sqlite3_stmt *stmt = NULL;

  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "SELECT f.id, f.folder, f.access_timestamp,"
    "       COUNT(i.id) AS image_count"
    " FROM main.film_rolls AS f"
    " LEFT JOIN main.images AS i ON i.film_id = f.id"
    " GROUP BY f.id"
    " ORDER BY f.access_timestamp DESC",
    -1, &stmt, NULL);
  // clang-format on

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "filmrolls");
  json_builder_begin_array(b);

  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    json_builder_begin_object(b);

    json_builder_set_member_name(b, "id");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 0));

    json_builder_set_member_name(b, "folder");
    const char *folder = (const char *)sqlite3_column_text(stmt, 1);
    json_builder_add_string_value(b, folder ? folder : "");

    json_builder_set_member_name(b, "access_timestamp");
    json_builder_add_int_value(b, sqlite3_column_int64(stmt, 2));

    json_builder_set_member_name(b, "image_count");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 3));

    json_builder_end_object(b);
  }
  sqlite3_finalize(stmt);

  json_builder_end_array(b);
  json_builder_end_object(b);

  JsonNode *rolls_result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, rolls_result);
  json_node_unref(rolls_result);
  g_object_unref(b);
  return resp;
}

char *dt_server_catalog_check_imported(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  if(!req->params || !json_object_has_member(req->params, "paths"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing paths parameter");

  JsonArray *paths = json_object_get_array_member(req->params, "paths");
  const guint len = json_array_get_length(paths);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "imported");
  json_builder_begin_array(b);

  for(guint i = 0; i < len; i++)
  {
    const char *fullpath = json_array_get_string_element(paths, i);
    if(!fullpath) continue;

    const dt_imgid_t id = dt_image_get_id_full_path(fullpath);
    if(dt_is_valid_imgid(id))
      json_builder_add_string_value(b, fullpath);
  }

  json_builder_end_array(b);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_catalog_import(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  if(!req->params || !json_object_has_member(req->params, "paths"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing paths parameter");

  JsonArray *paths = json_object_get_array_member(req->params, "paths");
  const guint len = json_array_get_length(paths);

  int imported = 0;
  int skipped = 0;
  dt_filmid_t last_filmid = NO_FILMID;

  for(guint i = 0; i < len; i++)
  {
    const char *fullpath = json_array_get_string_element(paths, i);
    if(!fullpath) continue;

    char *dirname = g_path_get_dirname(fullpath);

    dt_film_t film;
    dt_film_init(&film);
    const dt_filmid_t filmid = dt_film_new(&film, dirname);
    g_free(dirname);

    if(!dt_is_valid_filmid(filmid))
    {
      dt_film_cleanup(&film);
      skipped++;
      continue;
    }

    const dt_imgid_t imgid = dt_image_import(filmid, fullpath, FALSE, TRUE);
    dt_film_cleanup(&film);

    if(dt_is_valid_imgid(imgid))
    {
      imported++;
      last_filmid = filmid;
    }
    else
    {
      skipped++;
    }
  }

  // Signal that film rolls have been updated so the UI refreshes
  if(dt_is_valid_filmid(last_filmid))
    DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_FILMROLLS_IMPORTED, last_filmid);

  JsonBuilder *bi = json_builder_new();
  json_builder_begin_object(bi);
  json_builder_set_member_name(bi, "imported");
  json_builder_add_int_value(bi, imported);
  json_builder_set_member_name(bi, "skipped");
  json_builder_add_int_value(bi, skipped);
  json_builder_end_object(bi);

  JsonNode *iresult = json_builder_get_root(bi);
  char *iresp = dt_server_make_response(req->id, iresult);
  json_node_unref(iresult);
  g_object_unref(bi);
  return iresp;
}

char *dt_server_catalog_copy_import(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  if(!req->params || !json_object_has_member(req->params, "paths"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing paths parameter");

  JsonArray *paths = json_object_get_array_member(req->params, "paths");
  const guint len = json_array_get_length(paths);

  // Create an import session — uses session/base_directory_pattern,
  // session/sub_directory_pattern, session/filename_pattern from darktablerc
  struct dt_import_session_t *session = dt_import_session_new();

  int imported = 0;
  int skipped = 0;
  char *prev_filename = NULL;
  char *prev_output = NULL;

  for(guint i = 0; i < len; i++)
  {
    const char *src = json_array_get_string_element(paths, i);
    if(!src) continue;

    // Read source file into memory
    char *data = NULL;
    gsize size = 0;
    if(!g_file_get_contents(src, &data, &size, NULL))
    {
      dt_print(DT_DEBUG_ALWAYS, "[server] copy_import: failed to read `%s`", src);
      skipped++;
      continue;
    }

    struct stat statbuf;
    const int sts = stat(src, &statbuf);

    char *output = NULL;

    // If same basename as previous file (e.g. .NEF + .XMP sidecar),
    // reuse the output path, just change the extension
    if(prev_filename && prev_output && dt_has_same_path_basename(src, prev_filename))
    {
      output = dt_copy_filename_extension(prev_output, src);
    }
    else
    {
      // Extract basic EXIF info for variable expansion
      dt_image_basic_exif_t basic_exif = {0};
      dt_exif_get_basic_data((uint8_t *)data, size, &basic_exif);

      if(!basic_exif.datetime[0] && !sts)
      {
        // No EXIF datetime — fall back to file modification time
        dt_datetime_unix_to_exif(basic_exif.datetime,
                                 sizeof(basic_exif.datetime), &statbuf.st_mtime);
      }

      char *basename = g_path_get_basename(src);
      dt_import_session_set_exif_basic_info(session, &basic_exif);
      dt_import_session_set_filename(session, basename);

      const char *output_path = dt_import_session_path(session, FALSE);
      if(!output_path)
      {
        dt_print(DT_DEBUG_ALWAYS, "[server] copy_import: no session path for `%s`", src);
        g_free(basename);
        g_free(data);
        skipped++;
        continue;
      }

      const gboolean use_filename = dt_conf_get_bool("session/use_filename");
      const char *fname = dt_import_session_filename(session, use_filename);
      if(!fname)
      {
        dt_print(DT_DEBUG_ALWAYS, "[server] copy_import: no session filename for `%s`", src);
        g_free(basename);
        g_free(data);
        skipped++;
        continue;
      }

      output = g_build_filename(output_path, fname, NULL);
      g_free(basename);
    }

    // Write to destination
    if(!g_file_set_contents(output, data, size, NULL))
    {
      dt_print(DT_DEBUG_ALWAYS, "[server] copy_import: failed to write `%s`", output);
      g_free(data);
      g_free(output);
      skipped++;
      continue;
    }

    // Preserve original file timestamps
    if(!sts)
    {
      struct timeval times[2];
      times[0].tv_sec = statbuf.st_atime;
      times[1].tv_sec = statbuf.st_mtime;
#ifdef __APPLE__
#ifndef _POSIX_SOURCE
      times[0].tv_usec = statbuf.st_atimespec.tv_nsec / 1000;
      times[1].tv_usec = statbuf.st_mtimespec.tv_nsec / 1000;
#else
      times[0].tv_usec = statbuf.st_atimensec / 1000;
      times[1].tv_usec = statbuf.st_mtimensec / 1000;
#endif
#else
      times[0].tv_usec = statbuf.st_atim.tv_nsec / 1000;
      times[1].tv_usec = statbuf.st_mtim.tv_nsec / 1000;
#endif
      utimes(output, times);
    }

    g_free(data);

    // Import the copied file
    const dt_imgid_t imgid = dt_image_import(dt_import_session_film_id(session),
                                             output, FALSE, FALSE);
    if(!dt_is_valid_imgid(imgid))
    {
      dt_print(DT_DEBUG_ALWAYS, "[server] copy_import: dt_image_import failed for `%s`", output);
      g_free(output);
      skipped++;
      continue;
    }

    // Store metadata: image_id = "{original_filename}-{datetime}"
    // and PreservedFileName if the file was renamed during copy
    GFile *gfile = g_file_new_for_path(src);
    GFileInfo *info = g_file_query_info(gfile,
                                        G_FILE_ATTRIBUTE_STANDARD_NAME ","
                                        G_FILE_ATTRIBUTE_TIME_MODIFIED,
                                        G_FILE_QUERY_INFO_NONE, NULL, NULL);
    if(info)
    {
      const char *orig_name = g_file_info_get_name(info);
      const time_t mtime =
        g_file_info_get_attribute_uint64(info, G_FILE_ATTRIBUTE_TIME_MODIFIED);
      char dt_txt[DT_DATETIME_EXIF_LENGTH];
      dt_datetime_unix_to_exif(dt_txt, sizeof(dt_txt), &mtime);
      char *image_id = g_strconcat(orig_name, "-", dt_txt, NULL);
      dt_metadata_set(imgid, "Xmp.darktable.image_id", image_id, FALSE);

      // If file was renamed, preserve original filename
      gchar *output_basename = g_path_get_basename(output);
      if(g_strcmp0(output_basename, orig_name))
        dt_metadata_set(imgid, "Xmp.xmpMM.PreservedFileName", orig_name, FALSE);
      g_free(output_basename);
      g_free(image_id);
      g_object_unref(info);
    }
    g_object_unref(gfile);

    imported++;

    // Track previous filename/output for sidecar grouping
    g_free(prev_output);
    prev_output = output;
    prev_filename = (char *)src;  // points into JsonArray, valid for loop lifetime
  }

  g_free(prev_output);

  // Signal that film rolls have been updated
  const dt_filmid_t filmid = dt_import_session_film_id(session);
  if(dt_is_valid_filmid(filmid))
    DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_FILMROLLS_IMPORTED, filmid);

  dt_import_session_destroy(session);

  JsonBuilder *cb = json_builder_new();
  json_builder_begin_object(cb);
  json_builder_set_member_name(cb, "imported");
  json_builder_add_int_value(cb, imported);
  json_builder_set_member_name(cb, "skipped");
  json_builder_add_int_value(cb, skipped);
  json_builder_end_object(cb);

  JsonNode *ci_result = json_builder_get_root(cb);
  char *ci_resp = dt_server_make_response(req->id, ci_result);
  json_node_unref(ci_result);
  g_object_unref(cb);
  return ci_resp;
}

char *dt_server_catalog_get_file_thumbnail(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  if(!req->params || !json_object_has_member(req->params, "path"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing path parameter");

  const char *path = json_object_get_string_member(req->params, "path");

  // Extract embedded JPEG thumbnail from EXIF data
  uint8_t *buffer = NULL;
  size_t size = 0;
  char *mime_type = NULL;

  // dt_exif_get_thumbnail returns TRUE on error
  if(dt_exif_get_thumbnail(path, &buffer, &size, &mime_type))
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND,
                                "No embedded thumbnail found");

  // Base64-encode the JPEG data
  gchar *b64 = g_base64_encode(buffer, size);
  free(buffer);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "path");
  json_builder_add_string_value(b, path);

  json_builder_set_member_name(b, "mime");
  json_builder_add_string_value(b, mime_type ? mime_type : "image/jpeg");

  json_builder_set_member_name(b, "data");
  json_builder_add_string_value(b, b64);

  json_builder_end_object(b);

  g_free(b64);
  free(mime_type);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}
