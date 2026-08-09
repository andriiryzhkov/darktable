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

#include "common/lut_export_colorspace.h"
#include "common/darktable.h"
#include "common/file_location.h"

#include <json-glib/json-glib.h>
#include <math.h>
#include <string.h>

// id -> dt_lut_cs_t*, filled on first lookup and owned by this file
static GHashTable *_registry = NULL;
static GMutex _registry_lock;

// ---- transfer curves ----

float dt_lut_cs_encode(const dt_lut_cs_t *cs, const float linear)
{
  if(!cs) return linear;

  switch(cs->transfer)
  {
    case DT_LUT_CS_TRANSFER_LINEAR:
      return linear;

    case DT_LUT_CS_TRANSFER_GAMMA:
      // negatives have no real power-curve result; mirror the curve so the
      // grid stays monotonic through zero rather than collapsing
      return (linear < 0.0f)
        ? -powf(-linear, 1.0f / cs->exponent)
        : powf(linear, 1.0f / cs->exponent);

    case DT_LUT_CS_TRANSFER_SRGB:
      if(linear <= 0.0031308f) return 12.92f * linear;
      return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;

    case DT_LUT_CS_TRANSFER_REC709:
      if(linear < 0.018f) return 4.5f * linear;
      return 1.099f * powf(linear, 0.45f) - 0.099f;

    case DT_LUT_CS_TRANSFER_LOG:
      if(linear < cs->log_cut) return cs->log_e * linear + cs->log_f;
      return cs->log_c * log10f(linear + cs->log_b) + cs->log_d;
  }

  return linear;
}

float dt_lut_cs_decode(const dt_lut_cs_t *cs, const float code)
{
  if(!cs) return code;

  switch(cs->transfer)
  {
    case DT_LUT_CS_TRANSFER_LINEAR:
      return code;

    case DT_LUT_CS_TRANSFER_GAMMA:
      return (code < 0.0f)
        ? -powf(-code, cs->exponent)
        : powf(code, cs->exponent);

    case DT_LUT_CS_TRANSFER_SRGB:
      if(code <= 0.04045f) return code / 12.92f;
      return powf((code + 0.055f) / 1.055f, 2.4f);

    case DT_LUT_CS_TRANSFER_REC709:
      if(code < 0.081f) return code / 4.5f;
      return powf((code + 0.099f) / 1.099f, 1.0f / 0.45f);

    case DT_LUT_CS_TRANSFER_LOG:
      if(code < cs->log_cut_code) return (code - cs->log_f) / cs->log_e;
      return powf(10.0f, (code - cs->log_d) / cs->log_c) - cs->log_b;
  }

  return code;
}

// ---- JSON parsing ----

static gboolean _parse_transfer(JsonObject *obj, dt_lut_cs_t *cs)
{
  if(!json_object_has_member(obj, "transfer")) return FALSE;

  JsonObject *t = json_object_get_object_member(obj, "transfer");
  if(!t || !json_object_has_member(t, "type")) return FALSE;

  const char *type = json_object_get_string_member(t, "type");
  if(!type) return FALSE;

  if(!strcmp(type, "linear"))
  {
    cs->transfer = DT_LUT_CS_TRANSFER_LINEAR;
    return TRUE;
  }

  if(!strcmp(type, "srgb"))
  {
    cs->transfer = DT_LUT_CS_TRANSFER_SRGB;
    return TRUE;
  }

  if(!strcmp(type, "rec709"))
  {
    cs->transfer = DT_LUT_CS_TRANSFER_REC709;
    return TRUE;
  }

  if(!strcmp(type, "gamma"))
  {
    cs->transfer = DT_LUT_CS_TRANSFER_GAMMA;
    cs->exponent = json_object_has_member(t, "exponent")
      ? (float)json_object_get_double_member(t, "exponent")
      : 2.2f;
    return cs->exponent > 0.0f;
  }

  if(!strcmp(type, "log"))
  {
    // every field is required: a log curve with a defaulted coefficient
    // would be silently wrong rather than obviously broken
    static const char *const keys[] =
      { "cut", "cut_code", "b", "c", "d", "e", "f", NULL };

    for(int i = 0; keys[i]; i++)
      if(!json_object_has_member(t, keys[i]))
      {
        dt_print(DT_DEBUG_ALWAYS,
                 "[lut_colorspace] log transfer is missing `%s'", keys[i]);
        return FALSE;
      }

    cs->transfer = DT_LUT_CS_TRANSFER_LOG;
    cs->log_cut = (float)json_object_get_double_member(t, "cut");
    cs->log_cut_code = (float)json_object_get_double_member(t, "cut_code");
    cs->log_b = (float)json_object_get_double_member(t, "b");
    cs->log_c = (float)json_object_get_double_member(t, "c");
    cs->log_d = (float)json_object_get_double_member(t, "d");
    cs->log_e = (float)json_object_get_double_member(t, "e");
    cs->log_f = (float)json_object_get_double_member(t, "f");

    // c and e sit in denominators on the decode side
    return cs->log_c != 0.0f && cs->log_e != 0.0f;
  }

  dt_print(DT_DEBUG_ALWAYS,
           "[lut_colorspace] unknown transfer type `%s'", type);
  return FALSE;
}

static gboolean _parse_xy(JsonObject *obj, const char *key, float xy[2])
{
  if(!json_object_has_member(obj, key)) return FALSE;

  JsonArray *a = json_object_get_array_member(obj, key);
  if(!a || json_array_get_length(a) != 2) return FALSE;

  xy[0] = (float)json_array_get_double_element(a, 0);
  xy[1] = (float)json_array_get_double_element(a, 1);
  return TRUE;
}

static gboolean _parse_primaries(JsonObject *obj, dt_lut_cs_t *cs)
{
  if(!json_object_has_member(obj, "primaries")) return FALSE;

  JsonObject *p = json_object_get_object_member(obj, "primaries");
  if(!p) return FALSE;

  return _parse_xy(p, "red", cs->primaries[0])
    && _parse_xy(p, "green", cs->primaries[1])
    && _parse_xy(p, "blue", cs->primaries[2]);
}

static dt_lut_cs_t *_parse_entry(JsonObject *obj, const char *path)
{
  dt_lut_cs_t *cs = g_malloc0(sizeof(dt_lut_cs_t));

  if(json_object_has_member(obj, "id"))
    cs->id = g_strdup(json_object_get_string_member(obj, "id"));

  if(json_object_has_member(obj, "name"))
    cs->name = g_strdup(json_object_get_string_member(obj, "name"));

  const gboolean ok = cs->id && cs->id[0]
    && _parse_transfer(obj, cs)
    && _parse_primaries(obj, cs)
    && _parse_xy(obj, "white_point", cs->whitepoint);

  if(!ok)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_colorspace] %s: incomplete definition `%s', ignored",
             path, cs->id ? cs->id : "(no id)");
    g_free(cs->id);
    g_free(cs->name);
    g_free(cs);
    return NULL;
  }

  if(!cs->name) cs->name = g_strdup(cs->id);
  return cs;
}

static void _free_cs(gpointer data)
{
  dt_lut_cs_t *cs = data;
  if(!cs) return;
  g_free(cs->id);
  g_free(cs->name);
  g_free(cs);
}

// merge one colorspaces.json into the registry. entries are keyed by id, so
// a later file replaces individual entries rather than the whole set — that
// is what lets a user add one color space without copying the shipped list.
// missing files are normal: the user rarely has one
static void _load_file(const char *path)
{
  if(!g_file_test(path, G_FILE_TEST_EXISTS)) return;

  JsonParser *parser = json_parser_new();
  GError *err = NULL;

  if(!json_parser_load_from_file(parser, path, &err))
  {
    dt_print(DT_DEBUG_ALWAYS, "[lut_colorspace] %s: %s",
             path, err ? err->message : "cannot parse");
    g_clear_error(&err);
    g_object_unref(parser);
    return;
  }

  JsonNode *root = json_parser_get_root(parser);
  if(!root || !JSON_NODE_HOLDS_OBJECT(root))
  {
    dt_print(DT_DEBUG_ALWAYS, "[lut_colorspace] %s: not a JSON object", path);
    g_object_unref(parser);
    return;
  }

  JsonObject *top = json_node_get_object(root);

  // v1 is the frozen contract; a future schema lives under a parallel key
  if(json_object_has_member(top, "version"))
  {
    const int version = (int)json_object_get_int_member(top, "version");
    if(version != 1)
      dt_print(DT_DEBUG_ALWAYS,
               "[lut_colorspace] %s: version %d is not supported by this"
               " darktable; expected 1", path, version);
  }

  if(!json_object_has_member(top, "colorspaces"))
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_colorspace] %s: no `colorspaces' array", path);
    g_object_unref(parser);
    return;
  }

  JsonArray *arr = json_object_get_array_member(top, "colorspaces");
  const guint n = arr ? json_array_get_length(arr) : 0;

  for(guint i = 0; i < n; i++)
  {
    JsonObject *entry = json_array_get_object_element(arr, i);
    if(!entry) continue;

    dt_lut_cs_t *cs = _parse_entry(entry, path);
    if(cs) g_hash_table_replace(_registry, cs->id, cs);
  }

  g_object_unref(parser);
}

static void _ensure_registry(void)
{
  if(_registry) return;

  // the key is the entry's own id string, so only the value is freed
  _registry = g_hash_table_new_full(g_str_hash, g_str_equal, NULL, _free_cs);

  // shipped first, then the user's, so their entries win by id
  char datadir[PATH_MAX] = { 0 };
  dt_loc_get_datadir(datadir, sizeof(datadir));
  gchar *shipped =
    g_build_filename(datadir, "lut_export.json", NULL);
  _load_file(shipped);
  g_free(shipped);

  char configdir[PATH_MAX] = { 0 };
  dt_loc_get_user_config_dir(configdir, sizeof(configdir));
  gchar *user =
    g_build_filename(configdir, "lut_export.json", NULL);
  _load_file(user);
  g_free(user);

  dt_print(DT_DEBUG_ALWAYS, "[lut_colorspace] %u color spaces available",
           g_hash_table_size(_registry));
}

const dt_lut_cs_t *dt_lut_cs_find(const char *id)
{
  if(!id) return NULL;

  g_mutex_lock(&_registry_lock);
  _ensure_registry();
  const dt_lut_cs_t *cs = g_hash_table_lookup(_registry, id);
  g_mutex_unlock(&_registry_lock);

  if(!cs)
    dt_print(DT_DEBUG_ALWAYS, "[lut_colorspace] no color space `%s'", id);

  return cs;
}

GList *dt_lut_cs_list(void)
{
  g_mutex_lock(&_registry_lock);
  _ensure_registry();
  GList *l = g_hash_table_get_values(_registry);
  g_mutex_unlock(&_registry_lock);
  return l;
}

void dt_lut_cs_cleanup(void)
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
