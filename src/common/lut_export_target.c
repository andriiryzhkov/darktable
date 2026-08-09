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

#include "common/lut_export_target.h"
#include "common/darktable.h"
#include "common/file_location.h"
#include "common/lut_export_colorspace.h"

#include <json-glib/json-glib.h>
#include <string.h>

// id -> dt_lut_target_t*, filled on first lookup and owned by this file
static GHashTable *_registry = NULL;
static GMutex _registry_lock;

static void _free_target(gpointer data)
{
  dt_lut_target_t *t = data;
  if(!t) return;
  g_free(t->id);
  g_free(t->name);
  g_free(t->description);
  g_free(t->input_cs);
  g_free(t->output_cs);
  g_free(t);
}

static gboolean _parse_format(const char *s, dt_lut_export_format_t *format)
{
  if(!s || !strcmp(s, "cube"))
  {
    *format = DT_LUT_EXPORT_FORMAT_CUBE;
    return TRUE;
  }

  if(!strcmp(s, "hald_png"))
  {
    *format = DT_LUT_EXPORT_FORMAT_HALD_PNG;
    return TRUE;
  }

  return FALSE;
}

static dt_lut_target_t *_parse_entry(JsonObject *obj, const char *path)
{
  dt_lut_target_t *t = g_malloc0(sizeof(dt_lut_target_t));

  if(json_object_has_member(obj, "id"))
    t->id = g_strdup(json_object_get_string_member(obj, "id"));
  if(json_object_has_member(obj, "name"))
    t->name = g_strdup(json_object_get_string_member(obj, "name"));
  if(json_object_has_member(obj, "description"))
    t->description = g_strdup(json_object_get_string_member(obj, "description"));
  if(json_object_has_member(obj, "input_colorspace"))
    t->input_cs = g_strdup(json_object_get_string_member(obj, "input_colorspace"));
  if(json_object_has_member(obj, "output_colorspace"))
    t->output_cs = g_strdup(json_object_get_string_member(obj, "output_colorspace"));

  // a 33^3 grid is what camera LUT engines expect and what most .cube
  // consumers assume, so it is the one field worth defaulting
  t->grid_size = json_object_has_member(obj, "grid_size")
    ? (int)json_object_get_int_member(obj, "grid_size")
    : 33;

  const char *format = json_object_has_member(obj, "format")
    ? json_object_get_string_member(obj, "format")
    : NULL;

  const gboolean ok = t->id && t->id[0]
    && t->input_cs && t->output_cs
    && t->grid_size >= 2 && t->grid_size <= 256
    && _parse_format(format, &t->format);

  if(!ok)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_target] %s: incomplete target `%s', ignored",
             path, t->id ? t->id : "(no id)");
    _free_target(t);
    return NULL;
  }

  if(!t->name) t->name = g_strdup(t->id);
  return t;
}

// merge the `targets` array of one lut_export.json into the registry.
// entries are keyed by id, so a user file replaces individual targets rather
// than the whole set. missing files are normal: the user rarely has one
static void _load_file(const char *path)
{
  if(!g_file_test(path, G_FILE_TEST_EXISTS)) return;

  JsonParser *parser = json_parser_new();
  GError *err = NULL;

  if(!json_parser_load_from_file(parser, path, &err))
  {
    // the color space registry reads the same file and has already
    // complained about a parse error, so stay quiet here
    g_clear_error(&err);
    g_object_unref(parser);
    return;
  }

  JsonNode *root = json_parser_get_root(parser);
  if(!root || !JSON_NODE_HOLDS_OBJECT(root))
  {
    g_object_unref(parser);
    return;
  }

  JsonObject *top = json_node_get_object(root);
  if(!json_object_has_member(top, "targets"))
  {
    g_object_unref(parser);
    return;
  }

  JsonArray *arr = json_object_get_array_member(top, "targets");
  const guint n = arr ? json_array_get_length(arr) : 0;

  for(guint i = 0; i < n; i++)
  {
    JsonObject *entry = json_array_get_object_element(arr, i);
    if(!entry) continue;

    dt_lut_target_t *t = _parse_entry(entry, path);
    if(t) g_hash_table_replace(_registry, t->id, t);
  }

  g_object_unref(parser);
}

static void _ensure_registry(void)
{
  if(_registry) return;

  // the key is the entry's own id string, so only the value is freed
  _registry = g_hash_table_new_full(g_str_hash, g_str_equal, NULL, _free_target);

  // shipped first, then the user's, so their entries win by id
  char datadir[PATH_MAX] = { 0 };
  dt_loc_get_datadir(datadir, sizeof(datadir));
  gchar *shipped = g_build_filename(datadir, "lut_export.json", NULL);
  _load_file(shipped);
  g_free(shipped);

  char configdir[PATH_MAX] = { 0 };
  dt_loc_get_user_config_dir(configdir, sizeof(configdir));
  gchar *user = g_build_filename(configdir, "lut_export.json", NULL);
  _load_file(user);
  g_free(user);

  dt_print(DT_DEBUG_ALWAYS, "[lut_target] %u LUT targets available",
           g_hash_table_size(_registry));
}

// a target naming a color space that does not exist cannot be rendered.
// checked on lookup rather than at load time so that a user file supplying
// both a color space and a target that uses it works whichever order the
// two registries happen to be populated in
static gboolean _target_is_usable(const dt_lut_target_t *t)
{
  if(!dt_lut_cs_find(t->input_cs))
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_target] `%s' wants unknown input color space `%s'",
             t->id, t->input_cs);
    return FALSE;
  }

  if(!dt_lut_cs_find(t->output_cs))
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_target] `%s' wants unknown output color space `%s'",
             t->id, t->output_cs);
    return FALSE;
  }

  return TRUE;
}

const dt_lut_target_t *dt_lut_target_find(const char *id)
{
  if(!id) return NULL;

  g_mutex_lock(&_registry_lock);
  _ensure_registry();
  const dt_lut_target_t *t = g_hash_table_lookup(_registry, id);
  g_mutex_unlock(&_registry_lock);

  if(!t)
  {
    dt_print(DT_DEBUG_ALWAYS, "[lut_target] no target `%s'", id);
    return NULL;
  }

  return _target_is_usable(t) ? t : NULL;
}

static gint _by_name(gconstpointer a, gconstpointer b)
{
  const dt_lut_target_t *ta = a;
  const dt_lut_target_t *tb = b;
  return g_strcmp0(ta->name, tb->name);
}

GList *dt_lut_target_list(void)
{
  g_mutex_lock(&_registry_lock);
  _ensure_registry();
  GList *all = g_hash_table_get_values(_registry);
  g_mutex_unlock(&_registry_lock);

  GList *usable = NULL;
  for(GList *l = all; l; l = g_list_next(l))
    if(_target_is_usable(l->data)) usable = g_list_prepend(usable, l->data);

  g_list_free(all);
  return g_list_sort(usable, _by_name);
}

void dt_lut_target_cleanup(void)
{
  g_mutex_lock(&_registry_lock);
  if(_registry)
  {
    g_hash_table_destroy(_registry);
    _registry = NULL;
  }
  g_mutex_unlock(&_registry_lock);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
