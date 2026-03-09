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
#include "common/collection.h"
#include "common/database.h"
#include "common/datetime.h"
#include "common/debug.h"
#include "common/exif.h"
#include "common/film.h"
#include "common/grouping.h"
#include "common/history.h"
#include "common/image.h"
#include "common/image_cache.h"
#include "common/import_session.h"
#include "common/colorlabels.h"
#include "common/metadata.h"
#include "common/mipmap_cache.h"
#include "common/ratings.h"
#include "common/tags.h"
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
  const char *sort_field = NULL;
  const char *sort_order = NULL;

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
    if(json_object_has_member(req->params, "sort"))
      sort_field = json_object_get_string_member(req->params, "sort");
    if(json_object_has_member(req->params, "sort_order"))
      sort_order = json_object_get_string_member(req->params, "sort_order");
  }

  // Clamp limit
  if(limit < 1) limit = 1;
  if(limit > 1000) limit = 1000;

  // Check for collection rules (new multi-rule filter system)
  JsonArray *rules = NULL;
  if(req->params && json_object_has_member(req->params, "rules"))
  {
    JsonNode *rules_node = json_object_get_member(req->params, "rules");
    if(JSON_NODE_HOLDS_ARRAY(rules_node))
      rules = json_node_get_array(rules_node);
  }

  // Build the WHERE clause for collection rules
  GString *rules_where = NULL;
  if(rules && json_array_get_length(rules) > 0)
  {
    rules_where = g_string_new("");
    const guint n_rules = json_array_get_length(rules);
    for(guint r = 0; r < n_rules; r++)
    {
      JsonObject *rule = json_array_get_object_element(rules, r);
      if(!rule) continue;

      const char *mode = json_object_get_string_member_with_default(rule, "mode", "and");
      const char *prop = json_object_get_string_member_with_default(rule, "property", "");
      const char *text = json_object_get_string_member_with_default(rule, "text", "");
      if(!text[0]) continue;

      gchar *escaped = sqlite3_mprintf("%q", text);
      gchar *like_val = g_strdup_printf("%%%s%%", escaped);
      gchar *clause = NULL;

      if(!g_strcmp0(prop, "film_roll") || !g_strcmp0(prop, "folder"))
        clause = g_strdup_printf(
          "i.film_id IN (SELECT id FROM main.film_rolls WHERE folder LIKE '%s')", like_val);
      else if(!g_strcmp0(prop, "tag"))
        clause = g_strdup_printf(
          "i.id IN (SELECT imgid FROM main.tagged_images WHERE tagid IN"
          " (SELECT id FROM data.tags WHERE name LIKE '%s'))", like_val);
      else if(!g_strcmp0(prop, "camera"))
        clause = g_strdup_printf(
          "i.camera_id IN (SELECT id FROM main.cameras WHERE (maker || ' ' || model) LIKE '%s')", like_val);
      else if(!g_strcmp0(prop, "lens"))
        clause = g_strdup_printf(
          "i.lens_id IN (SELECT id FROM main.lens WHERE name LIKE '%s')", like_val);
      else if(!g_strcmp0(prop, "filename"))
        clause = g_strdup_printf("i.filename LIKE '%s'", like_val);
      else if(!g_strcmp0(prop, "rating"))
      {
        // supports: "unrated", "rejected", "3", ">=3", "<=2",
        // and comma-separated: "rejected,unrated,1,2,3"
        if(strchr(text, ','))
        {
          // Comma-separated list: build OR clause
          GString *or_clause = g_string_new("(");
          gchar **tokens = g_strsplit(text, ",", -1);
          gboolean or_first = TRUE;
          for(int t = 0; tokens[t]; t++)
          {
            g_strstrip(tokens[t]);
            if(!tokens[t][0]) continue;
            if(!or_first) g_string_append(or_clause, " OR ");
            or_first = FALSE;
            if(!g_ascii_strcasecmp(tokens[t], "rejected"))
              g_string_append(or_clause, "(i.flags & 7) = 6");
            else if(!g_ascii_strcasecmp(tokens[t], "unrated"))
              g_string_append(or_clause, "(i.flags & 7) = 0");
            else
            {
              int rv = atoi(tokens[t]);
              if(rv >= 1 && rv <= 5)
                g_string_append_printf(or_clause, "(i.flags & 7) = %d", rv);
            }
          }
          g_strfreev(tokens);
          g_string_append_c(or_clause, ')');
          if(!or_first)
            clause = g_string_free(or_clause, FALSE);
          else
            g_string_free(or_clause, TRUE);
        }
        else if(strstr(text, "unrated"))
          clause = g_strdup("(i.flags & 7) = 0");
        else if(strstr(text, "rejected"))
          clause = g_strdup("(i.flags & 7) = 6");
        else
        {
          const char *p = text;
          while(*p == ' ') p++;
          const char *op = "=";
          if(p[0] == '>' && p[1] == '=') { op = ">="; p += 2; }
          else if(p[0] == '<' && p[1] == '=') { op = "<="; p += 2; }
          else if(p[0] == '>') { op = ">"; p += 1; }
          else if(p[0] == '<') { op = "<"; p += 1; }
          while(*p == ' ') p++;
          int rating = atoi(p);
          if(rating >= 0 && rating <= 5)
            clause = g_strdup_printf("(i.flags & 7) %s %d AND (i.flags & 7) < 6", op, rating);
        }
      }
      else if(!g_strcmp0(prop, "color_label"))
      {
        // supports comma-separated: "red,green,blue"
        GString *color_set = g_string_new("");
        gchar **tokens = g_strsplit(text, ",", -1);
        for(int t = 0; tokens[t]; t++)
        {
          g_strstrip(tokens[t]);
          int color = -1;
          if(!g_ascii_strcasecmp(tokens[t], "red")) color = 0;
          else if(!g_ascii_strcasecmp(tokens[t], "yellow")) color = 1;
          else if(!g_ascii_strcasecmp(tokens[t], "green")) color = 2;
          else if(!g_ascii_strcasecmp(tokens[t], "blue")) color = 3;
          else if(!g_ascii_strcasecmp(tokens[t], "purple")) color = 4;
          if(color >= 0)
          {
            if(color_set->len > 0) g_string_append_c(color_set, ',');
            g_string_append_printf(color_set, "%d", color);
          }
        }
        g_strfreev(tokens);
        if(color_set->len > 0)
          clause = g_strdup_printf(
            "i.id IN (SELECT imgid FROM main.color_labels WHERE color IN (%s))",
            color_set->str);
        g_string_free(color_set, TRUE);
      }
      else if(!g_strcmp0(prop, "title") || !g_strcmp0(prop, "description") || !g_strcmp0(prop, "creator"))
        clause = g_strdup_printf(
          "i.id IN (SELECT md.id FROM main.meta_data AS md"
          " JOIN data.meta_data AS dmd ON md.key = dmd.key"
          " WHERE dmd.name = '%s' AND md.value LIKE '%s')", escaped, like_val);
      else if(!g_strcmp0(prop, "capture_date"))
        clause = g_strdup_printf(
          "SUBSTR(datetime(i.datetime_taken, 'unixepoch'), 1, 10) LIKE '%s'", like_val);
      else if(!g_strcmp0(prop, "import_time"))
        clause = g_strdup_printf(
          "i.import_timestamp > 0 AND SUBSTR(datetime(i.import_timestamp, 'unixepoch'), 1, 10) LIKE '%s'",
          like_val);
      else if(!g_strcmp0(prop, "change_time"))
        clause = g_strdup_printf(
          "i.change_timestamp > 0 AND SUBSTR(datetime(i.change_timestamp, 'unixepoch'), 1, 10) LIKE '%s'",
          like_val);
      else if(!g_strcmp0(prop, "aperture"))
        clause = g_strdup_printf(
          "('f/' || ROUND(i.aperture, 1)) LIKE '%s'", like_val);
      else if(!g_strcmp0(prop, "exposure"))
        clause = g_strdup_printf(
          "CASE WHEN i.exposure >= 1.0 THEN CAST(CAST(i.exposure AS INTEGER) AS TEXT) || 's'"
          " WHEN i.exposure > 0 THEN '1/' || CAST(ROUND(1.0/i.exposure) AS INTEGER) || 's'"
          " ELSE '0s' END LIKE '%s'", like_val);
      else if(!g_strcmp0(prop, "exposure_bias"))
        clause = g_strdup_printf(
          "(ROUND(i.exposure_bias, 1) || ' EV') LIKE '%s'", like_val);
      else if(!g_strcmp0(prop, "focal_length"))
        clause = g_strdup_printf(
          "(CAST(ROUND(i.focal_length) AS INTEGER) || 'mm') LIKE '%s'", like_val);
      else if(!g_strcmp0(prop, "iso"))
        clause = g_strdup_printf(
          "('ISO ' || CAST(ROUND(i.iso) AS INTEGER)) LIKE '%s'", like_val);
      else if(!g_strcmp0(prop, "aspect_ratio"))
        clause = g_strdup_printf(
          "CAST(ROUND(i.aspect_ratio, 2) AS TEXT) LIKE '%s'", like_val);
      else if(!g_strcmp0(prop, "white_balance"))
        clause = g_strdup_printf(
          "i.whitebalance_id IN (SELECT id FROM main.whitebalance WHERE name LIKE '%s')", like_val);
      else if(!g_strcmp0(prop, "flash"))
        clause = g_strdup_printf(
          "i.flash_id IN (SELECT id FROM main.flash WHERE name LIKE '%s')", like_val);
      else if(!g_strcmp0(prop, "exposure_program"))
        clause = g_strdup_printf(
          "i.exposure_program_id IN (SELECT id FROM main.exposure_program WHERE name LIKE '%s')", like_val);
      else if(!g_strcmp0(prop, "metering_mode"))
        clause = g_strdup_printf(
          "i.metering_mode_id IN (SELECT id FROM main.metering_mode WHERE name LIKE '%s')", like_val);
      else if(!g_strcmp0(prop, "group"))
        clause = g_strdup_printf(
          "i.group_id IN (SELECT id FROM main.images WHERE filename LIKE '%s')", like_val);
      else if(!g_strcmp0(prop, "history"))
      {
        if(strstr(text, "altered") && !strstr(text, "not"))
          clause = g_strdup("i.history_end > 0");
        else if(strstr(text, "not"))
          clause = g_strdup("(i.history_end IS NULL OR i.history_end = 0)");
      }

      sqlite3_free(escaped);
      g_free(like_val);

      if(!clause) continue;

      if(rules_where->len == 0)
      {
        // First rule: no prefix
        g_string_append_printf(rules_where, "(%s)", clause);
      }
      else
      {
        const char *sql_op = "AND";
        if(!g_strcmp0(mode, "or")) sql_op = "OR";
        else if(!g_strcmp0(mode, "and_not")) sql_op = "AND NOT";
        g_string_append_printf(rules_where, " %s (%s)", sql_op, clause);
      }
      g_free(clause);
    }
  }

  // Build query dynamically based on filters
  GString *query = g_string_new(
    "SELECT i.id, i.film_id, i.filename,"                                        /*  0- 2 */
    "       datetime(i.datetime_taken/1000000"
    "                + unixepoch('0001-01-01 00:00:00'), 'unixepoch'),"           /*  3    */
    "       i.flags, i.width, i.height, i.aspect_ratio,"                         /*  4- 7 */
    "       i.exposure, i.aperture, i.iso, i.focal_length,"                      /*  8-11 */
    "       f.folder,"                                                            /* 12    */
    "       i.group_id,"                                                          /* 13    */
    "       CASE WHEN i.history_end > 0 THEN 1 ELSE 0 END,"                     /* 14    */
    "       COALESCE((SELECT SUM(1 << cl.color) FROM main.color_labels AS cl"
    "                 WHERE cl.imgid = i.id), 0),"                               /* 15    */
    "       mk.name, md.name, ln.name,"                                          /* 16-18 */
    "       i.focus_distance, i.exposure_bias,"                                  /* 19-20 */
    "       i.longitude, i.latitude, i.altitude,"                                /* 21-23 */
    "       i.version, i.max_version,"                                           /* 24-25 */
    "       i.output_width, i.output_height,"                                    /* 26-27 */
    "       i.import_timestamp, i.change_timestamp,"
    "       i.export_timestamp, i.print_timestamp,"                              /* 28-31 */
    "       wb.name, fl.name, ep.name, mm.name,"                                /* 32-35 */
    "       i.crop, i.orientation"                                               /* 36-37 */
    " FROM main.images AS i"
    " LEFT JOIN main.film_rolls AS f ON i.film_id = f.id"
    " LEFT JOIN main.makers AS mk ON i.maker_id = mk.id"
    " LEFT JOIN main.models AS md ON i.model_id = md.id"
    " LEFT JOIN main.lens AS ln ON i.lens_id = ln.id"
    " LEFT JOIN main.whitebalance AS wb ON i.whitebalance_id = wb.id"
    " LEFT JOIN main.flash AS fl ON i.flash_id = fl.id"
    " LEFT JOIN main.exposure_program AS ep ON i.exposure_program_id = ep.id"
    " LEFT JOIN main.metering_mode AS mm ON i.metering_mode_id = mm.id");

  if(!rules_where && filter_tag_id >= 0)
    g_string_append(query,
      " JOIN main.tagged_images AS ti ON i.id = ti.imgid");

  g_string_append(query, " WHERE 1=1");

  if(rules_where && rules_where->len > 0)
  {
    g_string_append_printf(query, " AND (%s)", rules_where->str);
  }
  else
  {
    // Legacy individual filters (backward compatible)
    if(filter_film_id >= 0)
      g_string_append(query, " AND i.film_id = ?3");

    if(filter_tag_id >= 0)
      g_string_append(query, " AND ti.tagid = ?4");

    if(filter_rating_min >= 0)
      g_string_append_printf(query, " AND (i.flags & 7) >= ?5");

    if(filter_text && filter_text[0])
      g_string_append(query, " AND i.filename LIKE ?6");
  }

  // Build ORDER BY from sort params
  const char *order_col = "i.datetime_taken";
  const char *order_dir = "DESC";

  if(sort_field)
  {
    if(!g_strcmp0(sort_field, "filename"))              order_col = "i.filename";
    else if(!g_strcmp0(sort_field, "full path"))        order_col = "f.folder, i.filename";
    else if(!g_strcmp0(sort_field, "aspect ratio"))     order_col = "i.aspect_ratio";
    else if(!g_strcmp0(sort_field, "capture time"))     order_col = "i.datetime_taken";
    else if(!g_strcmp0(sort_field, "import time"))      order_col = "i.import_timestamp";
    else if(!g_strcmp0(sort_field, "modification time")) order_col = "i.change_timestamp";
    else if(!g_strcmp0(sort_field, "export time"))      order_col = "i.export_timestamp";
    else if(!g_strcmp0(sort_field, "print time"))       order_col = "i.print_timestamp";
    else if(!g_strcmp0(sort_field, "rating"))           order_col = "CASE WHEN i.flags & 8 = 8 THEN -1 ELSE i.flags & 7 END";
    else if(!g_strcmp0(sort_field, "color label"))      order_col = "i.color_labels";
    else if(!g_strcmp0(sort_field, "title"))            order_col = "i.filename"; // TODO: join metadata
    else if(!g_strcmp0(sort_field, "description"))      order_col = "i.filename"; // TODO: join metadata
    else if(!g_strcmp0(sort_field, "group"))            order_col = "i.group_id";
    else if(!g_strcmp0(sort_field, "id"))               order_col = "i.id";
    else if(!g_strcmp0(sort_field, "custom sort"))      order_col = "i.position";
    else if(!g_strcmp0(sort_field, "shuffle"))          order_col = "RANDOM()";
  }

  if(sort_order && !g_strcmp0(sort_order, "asc"))
    order_dir = "ASC";
  else if(sort_order && !g_strcmp0(sort_order, "desc"))
    order_dir = "DESC";

  g_string_append_printf(query, " ORDER BY %s %s LIMIT ?1 OFFSET ?2", order_col, order_dir);

  // Build matching count query
  GString *count_query = g_string_new(
    "SELECT COUNT(*)"
    " FROM main.images AS i");

  if(!rules_where && filter_tag_id >= 0)
    g_string_append(count_query,
      " JOIN main.tagged_images AS ti ON i.id = ti.imgid");

  g_string_append(count_query, " WHERE 1=1");

  if(rules_where && rules_where->len > 0)
  {
    g_string_append_printf(count_query, " AND (%s)", rules_where->str);
  }
  else
  {
    if(filter_film_id >= 0)
      g_string_append(count_query, " AND i.film_id = ?3");

    if(filter_tag_id >= 0)
      g_string_append(count_query, " AND ti.tagid = ?4");

    if(filter_rating_min >= 0)
      g_string_append(count_query, " AND (i.flags & 7) >= ?5");

    if(filter_text && filter_text[0])
      g_string_append(count_query, " AND i.filename LIKE ?6");
  }

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
    const char *dt_taken = (const char *)sqlite3_column_text(stmt, 3);
    json_builder_add_string_value(b, dt_taken ? dt_taken : "");

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

    // Derived fields
    int flags = sqlite3_column_int(stmt, 4);
    int raw_rating = flags & 7;
    json_builder_set_member_name(b, "rating");
    json_builder_add_int_value(b, raw_rating == 6 ? 0 : raw_rating);

    json_builder_set_member_name(b, "group_id");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 13));

    json_builder_set_member_name(b, "altered");
    json_builder_add_boolean_value(b, sqlite3_column_int(stmt, 14) != 0);

    json_builder_set_member_name(b, "local_copy");
    json_builder_add_boolean_value(b, (flags & DT_IMAGE_LOCAL_COPY) != 0);

    json_builder_set_member_name(b, "color_labels");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 15));

    // Lookup table names (maker, model, lens)
    const char *maker = (const char *)sqlite3_column_text(stmt, 16);
    json_builder_set_member_name(b, "maker");
    json_builder_add_string_value(b, maker ? maker : "");

    const char *model = (const char *)sqlite3_column_text(stmt, 17);
    json_builder_set_member_name(b, "model");
    json_builder_add_string_value(b, model ? model : "");

    const char *lens = (const char *)sqlite3_column_text(stmt, 18);
    json_builder_set_member_name(b, "lens");
    json_builder_add_string_value(b, lens ? lens : "");

    // Additional EXIF
    json_builder_set_member_name(b, "focus_distance");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 19));

    json_builder_set_member_name(b, "exposure_bias");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 20));

    // Geolocation
    json_builder_set_member_name(b, "longitude");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 21));

    json_builder_set_member_name(b, "latitude");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 22));

    json_builder_set_member_name(b, "altitude");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 23));

    // Version
    json_builder_set_member_name(b, "version");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 24));

    json_builder_set_member_name(b, "max_version");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 25));

    // Output dimensions
    json_builder_set_member_name(b, "output_width");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 26));

    json_builder_set_member_name(b, "output_height");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 27));

    // Timestamps (microseconds since 0001-01-01)
    json_builder_set_member_name(b, "import_timestamp");
    json_builder_add_int_value(b, sqlite3_column_int64(stmt, 28));

    json_builder_set_member_name(b, "change_timestamp");
    json_builder_add_int_value(b, sqlite3_column_int64(stmt, 29));

    json_builder_set_member_name(b, "export_timestamp");
    json_builder_add_int_value(b, sqlite3_column_int64(stmt, 30));

    json_builder_set_member_name(b, "print_timestamp");
    json_builder_add_int_value(b, sqlite3_column_int64(stmt, 31));

    // Lookup table names (whitebalance, flash, exposure_program, metering_mode)
    const char *wb = (const char *)sqlite3_column_text(stmt, 32);
    json_builder_set_member_name(b, "whitebalance");
    json_builder_add_string_value(b, wb ? wb : "");

    const char *flash_name = (const char *)sqlite3_column_text(stmt, 33);
    json_builder_set_member_name(b, "flash");
    json_builder_add_string_value(b, flash_name ? flash_name : "");

    const char *exp_prog = (const char *)sqlite3_column_text(stmt, 34);
    json_builder_set_member_name(b, "exposure_program");
    json_builder_add_string_value(b, exp_prog ? exp_prog : "");

    const char *meter = (const char *)sqlite3_column_text(stmt, 35);
    json_builder_set_member_name(b, "metering_mode");
    json_builder_add_string_value(b, meter ? meter : "");

    // Crop factor and orientation
    json_builder_set_member_name(b, "crop");
    json_builder_add_double_value(b, sqlite3_column_double(stmt, 36));

    json_builder_set_member_name(b, "orientation");
    json_builder_add_int_value(b, sqlite3_column_int(stmt, 37));

    // File size from disk
    json_builder_set_member_name(b, "file_size");
    if(folder && fname)
    {
      gchar *fullpath = g_build_filename(folder, fname, NULL);
      GStatBuf st;
      if(g_stat(fullpath, &st) == 0)
        json_builder_add_int_value(b, (gint64)st.st_size);
      else
        json_builder_add_int_value(b, -1);
      g_free(fullpath);
    }
    else
      json_builder_add_int_value(b, -1);

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

  if(!rules_where)
  {
    if(filter_film_id >= 0)
      DT_DEBUG_SQLITE3_BIND_INT(count_stmt, 3, filter_film_id);
    if(filter_tag_id >= 0)
      DT_DEBUG_SQLITE3_BIND_INT(count_stmt, 4, filter_tag_id);
    if(filter_rating_min >= 0)
      DT_DEBUG_SQLITE3_BIND_INT(count_stmt, 5, filter_rating_min);
    if(text_pattern)
      DT_DEBUG_SQLITE3_BIND_TEXT(count_stmt, 6, text_pattern, -1, SQLITE_TRANSIENT);
  }

  int total = 0;
  if(sqlite3_step(count_stmt) == SQLITE_ROW)
    total = sqlite3_column_int(count_stmt, 0);
  sqlite3_finalize(count_stmt);

  g_free(text_pattern);
  if(rules_where) g_string_free(rules_where, TRUE);

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

// Per-thumbnail result for parallel processing
typedef struct _thumb_result_t
{
  dt_imgid_t imgid;
  int width;
  int height;
  gchar *b64;       // base64 JPEG data (NULL on error)
  const char *error; // error string (NULL on success)
} _thumb_result_t;

typedef struct _thumb_job_t
{
  dt_imgid_t imgid;
  dt_mipmap_size_t mip;
  int quality;
  _thumb_result_t *result;
} _thumb_job_t;

static void _thumb_worker(gpointer data, gpointer user_data)
{
  (void)user_data;
  _thumb_job_t *job = data;
  _thumb_result_t *r = job->result;
  r->imgid = job->imgid;
  r->b64 = NULL;
  r->error = NULL;

  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(&buf, job->imgid, job->mip, DT_MIPMAP_BLOCKING, 'r');

  if(!buf.buf || buf.width <= 0 || buf.height <= 0)
  {
    dt_mipmap_cache_release(&buf);
    r->error = "not available";
    return;
  }

  const size_t jpeg_max_size = (size_t)buf.width * buf.height * 4 + 1024;
  uint8_t *jpeg_buf = g_malloc(jpeg_max_size);
  const int jpeg_size = dt_imageio_jpeg_compress(buf.buf, jpeg_buf,
                                                  buf.width, buf.height, job->quality);
  r->width = buf.width;
  r->height = buf.height;
  dt_mipmap_cache_release(&buf);

  if(jpeg_size <= 0)
  {
    g_free(jpeg_buf);
    r->error = "compression failed";
    return;
  }

  r->b64 = g_base64_encode(jpeg_buf, jpeg_size);
  g_free(jpeg_buf);
}

char *dt_server_catalog_get_thumbnails(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  if(!req->params || !json_object_has_member(req->params, "imgids"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  JsonArray *imgids_arr = json_object_get_array_member(req->params, "imgids");
  const guint count = json_array_get_length(imgids_arr);
  if(count == 0 || count > 200)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "imgids must have 1-200 entries");

  int requested_size = 720;
  if(json_object_has_member(req->params, "size"))
    requested_size = (int)json_object_get_int_member(req->params, "size");
  if(requested_size < 64) requested_size = 64;
  if(requested_size > 1920) requested_size = 1920;

  const dt_mipmap_size_t mip = dt_mipmap_cache_get_matching_size(requested_size, requested_size);

  // Process thumbnails in parallel using a thread pool
  _thumb_result_t *results = g_new0(_thumb_result_t, count);
  _thumb_job_t *jobs = g_new0(_thumb_job_t, count);

  GThreadPool *pool = g_thread_pool_new(_thumb_worker, NULL, MIN(count, 4), FALSE, NULL);

  for(guint i = 0; i < count; i++)
  {
    jobs[i].imgid = (dt_imgid_t)json_array_get_int_element(imgids_arr, i);
    jobs[i].mip = mip;
    jobs[i].quality = 85;
    jobs[i].result = &results[i];
    g_thread_pool_push(pool, &jobs[i], NULL);
  }

  // Wait for all threads to finish
  g_thread_pool_free(pool, FALSE, TRUE);

  // Build JSON response from collected results
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "thumbnails");
  json_builder_begin_array(b);

  for(guint i = 0; i < count; i++)
  {
    _thumb_result_t *r = &results[i];
    json_builder_begin_object(b);
    json_builder_set_member_name(b, "imgid");
    json_builder_add_int_value(b, r->imgid);

    if(r->b64)
    {
      json_builder_set_member_name(b, "width");
      json_builder_add_int_value(b, r->width);
      json_builder_set_member_name(b, "height");
      json_builder_add_int_value(b, r->height);
      json_builder_set_member_name(b, "data");
      json_builder_add_string_value(b, r->b64);
      g_free(r->b64);
    }
    else
    {
      json_builder_set_member_name(b, "error");
      json_builder_add_string_value(b, r->error ? r->error : "unknown error");
    }
    json_builder_end_object(b);
  }

  json_builder_end_array(b);
  json_builder_end_object(b);

  g_free(results);
  g_free(jobs);

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

/* ── Collection values endpoint ────────────────────────────────── */

char *dt_server_catalog_get_collection_values(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  if(!req->params || !json_object_has_member(req->params, "property"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing property parameter");

  const char *property = json_object_get_string_member(req->params, "property");
  const char *filter_raw = "";
  if(json_object_has_member(req->params, "filter"))
    filter_raw = json_object_get_string_member(req->params, "filter");

  gchar *like_pattern = (filter_raw && filter_raw[0])
    ? g_strdup_printf("%%%s%%", filter_raw)
    : g_strdup("%%");

  sqlite3_stmt *stmt = NULL;

  if(!g_strcmp0(property, "film_roll") || !g_strcmp0(property, "folder"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT f.id, f.folder, COUNT(i.id)"
      " FROM main.film_rolls AS f"
      " LEFT JOIN main.images AS i ON i.film_id = f.id"
      " WHERE f.folder LIKE ?1"
      " GROUP BY f.id"
      " ORDER BY f.folder",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "tag"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT t.id, t.name, COUNT(ti.imgid)"
      " FROM data.tags AS t"
      " LEFT JOIN main.tagged_images AS ti ON t.id = ti.tagid"
      " WHERE t.name LIKE ?1"
      "   AND t.name NOT LIKE 'darktable|%%'"
      " GROUP BY t.id"
      " ORDER BY t.name",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "camera"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT c.id, c.maker || ' ' || c.model, COUNT(i.id)"
      " FROM main.cameras AS c"
      " LEFT JOIN main.images AS i ON i.camera_id = c.id"
      " WHERE (c.maker || ' ' || c.model) LIKE ?1"
      " GROUP BY c.id"
      " ORDER BY c.maker, c.model",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "lens"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT l.id, l.name, COUNT(i.id)"
      " FROM main.lens AS l"
      " LEFT JOIN main.images AS i ON i.lens_id = l.id"
      " WHERE l.name LIKE ?1"
      " GROUP BY l.id"
      " ORDER BY l.name",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "filename"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT i.filename, i.filename, 1"
      " FROM main.images AS i"
      " WHERE i.filename LIKE ?1"
      " GROUP BY i.filename"
      " ORDER BY i.filename"
      " LIMIT 200",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "rating"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT (i.flags & 7) AS r,"
      " CASE (i.flags & 7)"
      "   WHEN 0 THEN 'unrated'"
      "   WHEN 6 THEN 'rejected'"
      "   ELSE (i.flags & 7) || ' star' || CASE WHEN (i.flags & 7) > 1 THEN 's' ELSE '' END"
      " END,"
      " COUNT(*)"
      " FROM main.images AS i"
      " GROUP BY r"
      " ORDER BY r",
      -1, &stmt, NULL);
    // clang-format on
    (void)like_pattern; // not filterable by text
  }
  else if(!g_strcmp0(property, "color_label"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT cl.color,"
      " CASE cl.color"
      "   WHEN 0 THEN 'red'"
      "   WHEN 1 THEN 'yellow'"
      "   WHEN 2 THEN 'green'"
      "   WHEN 3 THEN 'blue'"
      "   WHEN 4 THEN 'purple'"
      " END,"
      " COUNT(DISTINCT cl.imgid)"
      " FROM main.color_labels AS cl"
      " GROUP BY cl.color"
      " ORDER BY cl.color",
      -1, &stmt, NULL);
    // clang-format on
  }
  else if(!g_strcmp0(property, "title")
          || !g_strcmp0(property, "description")
          || !g_strcmp0(property, "creator"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT md.value, md.value, COUNT(DISTINCT md.id)"
      " FROM main.meta_data AS md"
      " JOIN data.meta_data AS dmd ON md.key = dmd.key"
      " WHERE dmd.name = ?2 AND md.value LIKE ?1"
      " GROUP BY md.value"
      " ORDER BY md.value"
      " LIMIT 200",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, property, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "capture_date"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT SUBSTR(datetime(i.datetime_taken, 'unixepoch'), 1, 10) AS d,"
      "        SUBSTR(datetime(i.datetime_taken, 'unixepoch'), 1, 10),"
      "        COUNT(*)"
      " FROM main.images AS i"
      " WHERE d LIKE ?1"
      " GROUP BY d"
      " ORDER BY d DESC"
      " LIMIT 200",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "import_time"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT SUBSTR(datetime(i.import_timestamp, 'unixepoch'), 1, 10) AS d,"
      "        SUBSTR(datetime(i.import_timestamp, 'unixepoch'), 1, 10),"
      "        COUNT(*)"
      " FROM main.images AS i"
      " WHERE i.import_timestamp > 0 AND d LIKE ?1"
      " GROUP BY d"
      " ORDER BY d DESC"
      " LIMIT 200",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "change_time"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT SUBSTR(datetime(i.change_timestamp, 'unixepoch'), 1, 10) AS d,"
      "        SUBSTR(datetime(i.change_timestamp, 'unixepoch'), 1, 10),"
      "        COUNT(*)"
      " FROM main.images AS i"
      " WHERE i.change_timestamp > 0 AND d LIKE ?1"
      " GROUP BY d"
      " ORDER BY d DESC"
      " LIMIT 200",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "aperture"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT i.aperture, 'f/' || ROUND(i.aperture, 1), COUNT(*)"
      " FROM main.images AS i"
      " WHERE i.aperture > 0"
      "   AND ('f/' || ROUND(i.aperture, 1)) LIKE ?1"
      " GROUP BY ROUND(i.aperture, 1)"
      " ORDER BY i.aperture",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "exposure"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT i.exposure,"
      " CASE"
      "   WHEN i.exposure >= 1.0 THEN CAST(CAST(i.exposure AS INTEGER) AS TEXT) || 's'"
      "   WHEN i.exposure > 0 THEN '1/' || CAST(ROUND(1.0/i.exposure) AS INTEGER) || 's'"
      "   ELSE '0s'"
      " END,"
      " COUNT(*)"
      " FROM main.images AS i"
      " WHERE i.exposure > 0"
      " GROUP BY ROUND(1.0/i.exposure)"
      " ORDER BY i.exposure DESC",
      -1, &stmt, NULL);
    // clang-format on
  }
  else if(!g_strcmp0(property, "exposure_bias"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT i.exposure_bias,"
      " ROUND(i.exposure_bias, 1) || ' EV',"
      " COUNT(*)"
      " FROM main.images AS i"
      " GROUP BY ROUND(i.exposure_bias, 1)"
      " ORDER BY i.exposure_bias",
      -1, &stmt, NULL);
    // clang-format on
  }
  else if(!g_strcmp0(property, "focal_length"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT i.focal_length,"
      " CAST(ROUND(i.focal_length) AS INTEGER) || 'mm',"
      " COUNT(*)"
      " FROM main.images AS i"
      " WHERE i.focal_length > 0"
      "   AND (CAST(ROUND(i.focal_length) AS INTEGER) || 'mm') LIKE ?1"
      " GROUP BY ROUND(i.focal_length)"
      " ORDER BY i.focal_length",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "iso"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT i.iso,"
      " 'ISO ' || CAST(ROUND(i.iso) AS INTEGER),"
      " COUNT(*)"
      " FROM main.images AS i"
      " WHERE i.iso > 0"
      "   AND ('ISO ' || CAST(ROUND(i.iso) AS INTEGER)) LIKE ?1"
      " GROUP BY ROUND(i.iso)"
      " ORDER BY i.iso",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "aspect_ratio"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT ROUND(i.aspect_ratio, 2),"
      " ROUND(i.aspect_ratio, 2),"
      " COUNT(*)"
      " FROM main.images AS i"
      " WHERE i.aspect_ratio > 0"
      " GROUP BY ROUND(i.aspect_ratio, 2)"
      " ORDER BY i.aspect_ratio",
      -1, &stmt, NULL);
    // clang-format on
  }
  else if(!g_strcmp0(property, "white_balance"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT w.id, w.name, COUNT(i.id)"
      " FROM main.whitebalance AS w"
      " LEFT JOIN main.images AS i ON i.whitebalance_id = w.id"
      " WHERE w.name LIKE ?1"
      " GROUP BY w.id"
      " ORDER BY w.name",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "flash"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT f.id, f.name, COUNT(i.id)"
      " FROM main.flash AS f"
      " LEFT JOIN main.images AS i ON i.flash_id = f.id"
      " WHERE f.name LIKE ?1"
      " GROUP BY f.id"
      " ORDER BY f.name",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "exposure_program"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT ep.id, ep.name, COUNT(i.id)"
      " FROM main.exposure_program AS ep"
      " LEFT JOIN main.images AS i ON i.exposure_program_id = ep.id"
      " WHERE ep.name LIKE ?1"
      " GROUP BY ep.id"
      " ORDER BY ep.name",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "metering_mode"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT mm.id, mm.name, COUNT(i.id)"
      " FROM main.metering_mode AS mm"
      " LEFT JOIN main.images AS i ON i.metering_mode_id = mm.id"
      " WHERE mm.name LIKE ?1"
      " GROUP BY mm.id"
      " ORDER BY mm.name",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "group"))
  {
    // Show group leaders that have grouped images
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT i2.group_id, i2.filename, COUNT(*)"
      " FROM main.images AS i2"
      " WHERE i2.id != i2.group_id"
      "   AND i2.filename LIKE ?1"
      " GROUP BY i2.group_id"
      " ORDER BY COUNT(*) DESC"
      " LIMIT 200",
      -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, like_pattern, -1, SQLITE_TRANSIENT);
  }
  else if(!g_strcmp0(property, "history"))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT CASE WHEN i.history_end > 0 THEN 1 ELSE 0 END AS altered,"
      " CASE WHEN i.history_end > 0 THEN 'altered' ELSE 'not altered' END,"
      " COUNT(*)"
      " FROM main.images AS i"
      " GROUP BY altered"
      " ORDER BY altered",
      -1, &stmt, NULL);
    // clang-format on
  }
  else
  {
    g_free(like_pattern);
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Unknown property");
  }

  JsonBuilder *cv = json_builder_new();
  json_builder_begin_object(cv);
  json_builder_set_member_name(cv, "values");
  json_builder_begin_array(cv);

  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    json_builder_begin_object(cv);

    json_builder_set_member_name(cv, "id");
    if(sqlite3_column_type(stmt, 0) == SQLITE_INTEGER)
      json_builder_add_int_value(cv, sqlite3_column_int(stmt, 0));
    else
    {
      const char *id_str = (const char *)sqlite3_column_text(stmt, 0);
      json_builder_add_string_value(cv, id_str ? id_str : "");
    }

    json_builder_set_member_name(cv, "label");
    const char *label = (const char *)sqlite3_column_text(stmt, 1);
    json_builder_add_string_value(cv, label ? label : "");

    json_builder_set_member_name(cv, "count");
    json_builder_add_int_value(cv, sqlite3_column_int(stmt, 2));

    json_builder_end_object(cv);
  }
  sqlite3_finalize(stmt);
  g_free(like_pattern);

  json_builder_end_array(cv);
  json_builder_end_object(cv);

  JsonNode *cv_result = json_builder_get_root(cv);
  char *cv_resp = dt_server_make_response(req->id, cv_result);
  json_node_unref(cv_result);
  g_object_unref(cv);
  return cv_resp;
}


/* ── Helper: parse imgids array from params ─────────────────────── */

static GList *_parse_imgids(const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "imgids"))
    return NULL;

  JsonArray *arr = json_object_get_array_member(req->params, "imgids");
  const guint len = json_array_get_length(arr);
  GList *ids = NULL;
  for(guint i = 0; i < len; i++)
    ids = g_list_prepend(ids, GINT_TO_POINTER((int)json_array_get_int_element(arr, i)));
  return g_list_reverse(ids);
}

static char *_make_count_response(const dt_server_request_t *req, int count)
{
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "count");
  json_builder_add_int_value(b, count);
  json_builder_end_object(b);
  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}


/* ── catalog.image_remove ───────────────────────────────────────── */

char *dt_server_catalog_image_remove(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    dt_image_remove(imgid);
    count++;
  }

  dt_film_remove_empty();
  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF, ids);
  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_FILMROLLS_CHANGED);

  return _make_count_response(req, count);
}


/* ── catalog.image_delete ───────────────────────────────────────── */

char *dt_server_catalog_image_delete(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  const gboolean send_to_trash = dt_conf_get_bool("send_to_trash");
  int count = 0;

  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    char pathname[PATH_MAX] = { 0 };
    gboolean from_cache = TRUE;
    dt_image_full_path(imgid, pathname, sizeof(pathname), &from_cache);

    dt_image_remove(imgid);

    if(pathname[0])
    {
      if(send_to_trash)
      {
        GFile *gfile = g_file_new_for_path(pathname);
        g_file_trash(gfile, NULL, NULL);
        g_object_unref(gfile);
      }
      else
      {
        g_unlink(pathname);
      }
      // Also remove XMP sidecar if present
      char *xmp = g_strconcat(pathname, ".xmp", NULL);
      if(g_file_test(xmp, G_FILE_TEST_EXISTS))
      {
        if(send_to_trash)
        {
          GFile *gxmp = g_file_new_for_path(xmp);
          g_file_trash(gxmp, NULL, NULL);
          g_object_unref(gxmp);
        }
        else
          g_unlink(xmp);
      }
      g_free(xmp);
    }
    count++;
  }

  dt_film_remove_empty();
  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF, ids);
  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_FILMROLLS_CHANGED);

  return _make_count_response(req, count);
}


/* ── catalog.image_duplicate ────────────────────────────────────── */

char *dt_server_catalog_image_duplicate(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    const dt_imgid_t newid = dt_image_duplicate(imgid);
    if(dt_is_valid_imgid(newid))
    {
      dt_history_copy_and_paste_on_image(imgid, newid, FALSE, NULL, TRUE, TRUE, TRUE);
      dt_image_cache_set_change_timestamp_from_image(newid, imgid);
      count++;
    }
  }

  g_list_free(ids);
  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_FILMROLLS_CHANGED);

  return _make_count_response(req, count);
}


/* ── catalog.image_rotate ───────────────────────────────────────── */

char *dt_server_catalog_image_rotate(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  /* cw: 0 = clockwise, 1 = counter-clockwise, 2 = reset */
  int cw = 0;
  if(req->params && json_object_has_member(req->params, "direction"))
    cw = (int)json_object_get_int_member(req->params, "direction");

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    dt_image_flip(imgid, cw);
    dt_image_set_aspect_ratio(imgid, FALSE);
    dt_mipmap_cache_remove(imgid);
    count++;
  }

  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_ASPECT_RATIO,
                             g_list_copy(ids));
  g_list_free(ids);

  return _make_count_response(req, count);
}


/* ── catalog.image_group ────────────────────────────────────────── */

char *dt_server_catalog_image_group(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  /* Use first image as group leader */
  const dt_imgid_t group_id = GPOINTER_TO_INT(ids->data);
  int count = 0;

  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    dt_grouping_add_to_group(group_id, imgid);
    count++;
  }

  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF,
                             ids);

  return _make_count_response(req, count);
}


/* ── catalog.image_ungroup ──────────────────────────────────────── */

char *dt_server_catalog_image_ungroup(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    const dt_imgid_t new_group = dt_grouping_remove_from_group(imgid);
    if(dt_is_valid_imgid(new_group))
      count++;
  }

  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF,
                             ids);

  return _make_count_response(req, count);
}


/* ── catalog.image_copy_local ───────────────────────────────────── */

char *dt_server_catalog_image_copy_local(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    dt_image_local_copy_set(imgid);
    count++;
  }

  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF,
                             ids);

  return _make_count_response(req, count);
}


/* ── catalog.image_resync_local ─────────────────────────────────── */

char *dt_server_catalog_image_resync_local(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    dt_image_local_copy_reset(imgid);
    count++;
  }

  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF,
                             ids);

  return _make_count_response(req, count);
}


/* ── catalog.image_refresh_exif ─────────────────────────────────── */

char *dt_server_catalog_image_refresh_exif(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    gboolean from_cache = TRUE;
    char sourcefile[PATH_MAX];
    dt_image_full_path(imgid, sourcefile, sizeof(sourcefile), &from_cache);

    dt_image_t *img = dt_image_cache_get(imgid, 'w');
    if(img)
    {
      img->job_flags |= DT_IMAGE_JOB_NO_METADATA;
      dt_exif_read(img, sourcefile);
      dt_image_cache_write_release_info(img, DT_IMAGE_CACHE_SAFE,
                                        "dt_server_catalog_image_refresh_exif");
      count++;
    }
  }

  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF,
                             ids);
  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_TAG_CHANGED);

  return _make_count_response(req, count);
}


/* ── catalog.metadata_paste ─────────────────────────────────────── */
/* Copies metadata from source_imgid to target imgids.
 * Params: { source_imgid, imgids, flags: { ratings, colors, tags, geotags, metadata }, mode: "merge"|"overwrite" }
 */

char *dt_server_catalog_metadata_paste(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  if(!req->params || !json_object_has_member(req->params, "source_imgid")
     || !json_object_has_member(req->params, "imgids"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                "Missing source_imgid or imgids parameter");

  const dt_imgid_t src = (dt_imgid_t)json_object_get_int_member(req->params, "source_imgid");
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Empty imgids");

  /* Parse flags */
  gboolean f_ratings = TRUE, f_colors = TRUE, f_tags = TRUE, f_geotags = TRUE, f_metadata = TRUE;
  if(json_object_has_member(req->params, "flags"))
  {
    JsonObject *flags = json_object_get_object_member(req->params, "flags");
    if(json_object_has_member(flags, "ratings"))
      f_ratings = json_object_get_boolean_member(flags, "ratings");
    if(json_object_has_member(flags, "colors"))
      f_colors = json_object_get_boolean_member(flags, "colors");
    if(json_object_has_member(flags, "tags"))
      f_tags = json_object_get_boolean_member(flags, "tags");
    if(json_object_has_member(flags, "geotags"))
      f_geotags = json_object_get_boolean_member(flags, "geotags");
    if(json_object_has_member(flags, "metadata"))
      f_metadata = json_object_get_boolean_member(flags, "metadata");
  }

  const gboolean clear_on = json_object_has_member(req->params, "mode")
    && g_strcmp0(json_object_get_string_member(req->params, "mode"), "overwrite") == 0;

  int count = 0;

  if(f_ratings)
  {
    const int stars = dt_ratings_get(src);
    dt_ratings_apply_on_list(ids, stars, TRUE);
    count++;
  }

  if(f_colors)
  {
    const int labels = dt_colorlabels_get_labels(src);
    if(clear_on)
    {
      /* Clear then set each label */
      for(GList *t = ids; t; t = g_list_next(t))
      {
        const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
        dt_colorlabels_set_labels(g_list_prepend(NULL, GINT_TO_POINTER(imgid)),
                                  0, TRUE, FALSE);
      }
    }
    for(int c = 0; c < 5; c++)
    {
      if(labels & (1 << c))
        dt_colorlabels_set_labels(g_list_copy(ids), c, FALSE, FALSE);
    }
    count++;
  }

  if(f_tags)
  {
    GList *tag_list = NULL;
    const guint tag_count = dt_tag_get_attached(src, &tag_list, TRUE);
    if(tag_count > 0)
    {
      dt_tag_set_tags(tag_list, g_list_copy(ids), TRUE, clear_on, FALSE);
    }
    g_list_free_full(tag_list, g_free);
    count++;
  }

  if(f_geotags)
  {
    dt_image_geoloc_t geoloc;
    dt_image_get_location(src, &geoloc);
    dt_image_set_locations(g_list_copy(ids), &geoloc, FALSE);
    count++;
  }

  if(f_metadata)
  {
    /* Copy all dt metadata keys from source, apply to targets */
    GList *md_list = dt_metadata_get_list();
    GList *kv = NULL;
    for(GList *m = md_list; m; m = g_list_next(m))
    {
      const dt_metadata_t *md = m->data;
      uint32_t cnt = 0;
      GList *val = dt_metadata_get(src, md->tagname, &cnt);
      kv = g_list_prepend(kv, g_strdup(md->tagname));
      kv = g_list_prepend(kv, val ? g_strdup(val->data) : g_strdup(""));
      g_list_free_full(val, g_free);
    }
    kv = g_list_reverse(kv);
    dt_metadata_set_list(g_list_copy(ids), kv, FALSE);
    count++;
  }

  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_TAG_CHANGED);
  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_IMAGE_INFO_CHANGED, g_list_copy(ids));
  g_list_free(ids);

  return _make_count_response(req, count);
}


/* ── catalog.metadata_clear ─────────────────────────────────────── */

char *dt_server_catalog_metadata_clear(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  gboolean f_ratings = TRUE, f_colors = TRUE, f_tags = TRUE, f_geotags = TRUE, f_metadata = TRUE;
  if(req->params && json_object_has_member(req->params, "flags"))
  {
    JsonObject *flags = json_object_get_object_member(req->params, "flags");
    if(json_object_has_member(flags, "ratings"))
      f_ratings = json_object_get_boolean_member(flags, "ratings");
    if(json_object_has_member(flags, "colors"))
      f_colors = json_object_get_boolean_member(flags, "colors");
    if(json_object_has_member(flags, "tags"))
      f_tags = json_object_get_boolean_member(flags, "tags");
    if(json_object_has_member(flags, "geotags"))
      f_geotags = json_object_get_boolean_member(flags, "geotags");
    if(json_object_has_member(flags, "metadata"))
      f_metadata = json_object_get_boolean_member(flags, "metadata");
  }

  int count = 0;

  if(f_ratings)
  {
    dt_ratings_apply_on_list(g_list_copy(ids), 0, FALSE);
    count++;
  }

  if(f_colors)
  {
    for(GList *t = ids; t; t = g_list_next(t))
    {
      const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
      dt_colorlabels_set_labels(g_list_prepend(NULL, GINT_TO_POINTER(imgid)),
                                0, TRUE, FALSE);
    }
    count++;
  }

  if(f_tags)
  {
    dt_tag_set_tags(NULL, g_list_copy(ids), TRUE, TRUE, FALSE);
    count++;
  }

  if(f_geotags)
  {
    dt_image_geoloc_t empty = { NAN, NAN, NAN };
    dt_image_set_locations(g_list_copy(ids), &empty, FALSE);
    count++;
  }

  if(f_metadata)
  {
    GList *md_list = dt_metadata_get_list();
    GList *kv = NULL;
    for(GList *m = md_list; m; m = g_list_next(m))
    {
      const dt_metadata_t *md = m->data;
      kv = g_list_prepend(kv, g_strdup(md->tagname));
      kv = g_list_prepend(kv, g_strdup(""));
    }
    kv = g_list_reverse(kv);
    dt_metadata_set_list(g_list_copy(ids), kv, FALSE);
    count++;
  }

  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_TAG_CHANGED);
  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_IMAGE_INFO_CHANGED, g_list_copy(ids));
  g_list_free(ids);

  return _make_count_response(req, count);
}


/* ── catalog.image_set_monochrome ───────────────────────────────── */

char *dt_server_catalog_image_set_monochrome(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  /* monochrome: 0 = set color, 2 = set monochrome */
  int mode = 2;
  if(req->params && json_object_has_member(req->params, "monochrome"))
    mode = json_object_get_boolean_member(req->params, "monochrome") ? 2 : 0;

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    dt_image_set_monochrome_flag(imgid, mode == 2);
    count++;
  }

  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF,
                             ids);

  return _make_count_response(req, count);
}


/* ── catalog.image_move ────────────────────────────────────────── */

char *dt_server_catalog_image_move(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  if(!req->params || !json_object_has_member(req->params, "path"))
  {
    g_list_free(ids);
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing path parameter");
  }

  const char *path = json_object_get_string_member(req->params, "path");

  /* Get or create film roll for destination directory */
  dt_film_t film;
  dt_film_init(&film);
  const dt_filmid_t filmid = dt_film_new(&film, path);
  if(!dt_is_valid_filmid(filmid))
  {
    dt_film_cleanup(&film);
    g_list_free(ids);
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Cannot access destination folder");
  }

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    if(!dt_image_move(imgid, filmid))
      count++;
  }

  dt_film_cleanup(&film);
  dt_film_remove_empty();
  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF, ids);
  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_FILMROLLS_CHANGED);

  return _make_count_response(req, count);
}


/* ── catalog.image_copy_to ─────────────────────────────────────── */

char *dt_server_catalog_image_copy_to(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  GList *ids = _parse_imgids(req);
  if(!ids)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgids parameter");

  if(!req->params || !json_object_has_member(req->params, "path"))
  {
    g_list_free(ids);
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing path parameter");
  }

  const char *path = json_object_get_string_member(req->params, "path");

  dt_film_t film;
  dt_film_init(&film);
  const dt_filmid_t filmid = dt_film_new(&film, path);
  if(!dt_is_valid_filmid(filmid))
  {
    dt_film_cleanup(&film);
    g_list_free(ids);
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Cannot access destination folder");
  }

  int count = 0;
  for(GList *t = ids; t; t = g_list_next(t))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(t->data);
    const dt_imgid_t new_imgid = dt_image_copy(imgid, filmid);
    if(dt_is_valid_imgid(new_imgid))
      count++;
  }

  dt_film_cleanup(&film);
  dt_collection_update_query(darktable.collection,
                             DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF, ids);
  DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_FILMROLLS_CHANGED);

  return _make_count_response(req, count);
}
