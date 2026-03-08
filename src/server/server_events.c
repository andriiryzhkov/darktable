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

/*
 * Event bridge: connects darktable's GSignal-based signal system to
 * the JSON event system used by the webview UI.
 *
 * Each signal callback builds a JsonNode with relevant data and calls
 * dt_server_queue_event(), which either pushes to the event queue
 * (IPC mode) or calls the event callback directly (embedded mode).
 */

#include "server/server.h"
#include "common/darktable.h"
#include "common/collection.h"
#include "control/signal.h"

#include <json-glib/json-glib.h>


/* ── Signal callbacks ─────────────────────────────────────────────── */

static void _on_collection_changed(gpointer instance,
                                   const dt_collection_change_t query_change,
                                   const dt_collection_properties_t changed_property,
                                   gpointer imgs,
                                   const int next,
                                   gpointer user_data)
{
  dt_server_t *server = user_data;

  JsonNode *data = json_node_alloc();
  JsonObject *obj = json_object_new();
  json_object_set_int_member(obj, "change_type", (gint64)query_change);
  json_node_init_object(data, obj);
  json_object_unref(obj);

  dt_server_queue_event(server, "collection.changed", data);
  json_node_unref(data);
}

static void _on_image_imported(gpointer instance,
                               const dt_imgid_t imgid,
                               gpointer user_data)
{
  dt_server_t *server = user_data;

  JsonNode *data = json_node_alloc();
  JsonObject *obj = json_object_new();
  json_object_set_int_member(obj, "imgid", (gint64)imgid);
  json_node_init_object(data, obj);
  json_object_unref(obj);

  dt_server_queue_event(server, "image.imported", data);
  json_node_unref(data);
}

static void _on_mipmap_updated(gpointer instance,
                               const dt_imgid_t imgid,
                               gpointer user_data)
{
  dt_server_t *server = user_data;

  JsonNode *data = json_node_alloc();
  JsonObject *obj = json_object_new();
  json_object_set_int_member(obj, "imgid", (gint64)imgid);
  json_node_init_object(data, obj);
  json_object_unref(obj);

  dt_server_queue_event(server, "image.thumbnail_ready", data);
  json_node_unref(data);
}

static void _on_preview_pipe_finished(gpointer instance,
                                      gpointer user_data)
{
  dt_server_t *server = user_data;
  dt_server_queue_event(server, "develop.preview_ready", NULL);
}

static void _on_history_changed(gpointer instance,
                                gpointer user_data)
{
  dt_server_t *server = user_data;
  dt_server_queue_event(server, "develop.history_changed", NULL);
}


/* ── Init / cleanup ───────────────────────────────────────────────── */

void dt_server_events_init(dt_server_t *server)
{
  DT_CONTROL_SIGNAL_CONNECT(DT_SIGNAL_COLLECTION_CHANGED,
                            _on_collection_changed, server);
  DT_CONTROL_SIGNAL_CONNECT(DT_SIGNAL_IMAGE_IMPORT,
                            _on_image_imported, server);
  DT_CONTROL_SIGNAL_CONNECT(DT_SIGNAL_DEVELOP_MIPMAP_UPDATED,
                            _on_mipmap_updated, server);
  DT_CONTROL_SIGNAL_CONNECT(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED,
                            _on_preview_pipe_finished, server);
  DT_CONTROL_SIGNAL_CONNECT(DT_SIGNAL_DEVELOP_HISTORY_CHANGE,
                            _on_history_changed, server);

  fprintf(stderr, "[server] event bridge initialized (5 signals connected)\n");
}

void dt_server_events_cleanup(dt_server_t *server)
{
  DT_CONTROL_SIGNAL_DISCONNECT(_on_collection_changed, server);
  DT_CONTROL_SIGNAL_DISCONNECT(_on_image_imported, server);
  DT_CONTROL_SIGNAL_DISCONNECT(_on_mipmap_updated, server);
  DT_CONTROL_SIGNAL_DISCONNECT(_on_preview_pipe_finished, server);
  DT_CONTROL_SIGNAL_DISCONNECT(_on_history_changed, server);

  fprintf(stderr, "[server] event bridge cleaned up\n");
}
