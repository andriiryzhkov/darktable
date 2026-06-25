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

#include "common/ai/embeddings.h"
#include "ai/backend.h"
#include "common/ai_models.h"
#include "common/darktable.h"
#include "common/database.h"
#include "common/debug.h"
#include "common/file_location.h"
#include "common/mipmap_cache.h"
#include "common/tags.h"
#include "control/conf.h"
#include "control/jobs.h"
#include "control/signal.h"
#include "sqlite-vec.h"
#include <json-glib/json-glib.h>

#include <math.h>
#include <string.h>

#define EMBED_MODEL_TASK "embed"
#define EMBED_DB_VERSION 1
#define TAG_MIN_EXAMPLES 3

// embeddings DB is attached to the main database handle as "embed"
static gboolean _embed_attached = FALSE;

// forward decl: definition lives next to the tag-centroid helpers below
static void _normalize(float *vec, int dim);

// upgrade the embeddings database schema step by step.
// returns the new version, or -1 on error
static int _upgrade_embed_schema(sqlite3 *db, int from_version)
{
  if(from_version == EMBED_DB_VERSION)
    return from_version;

  if(from_version > EMBED_DB_VERSION)
  {
    dt_print(DT_DEBUG_AI,
             "[embeddings] database version %d is newer than "
             "supported %d — cannot downgrade",
             from_version, EMBED_DB_VERSION);
    return -1;
  }

  // version 0 → 1: initial schema
  if(from_version == 0)
  {
    int rc = sqlite3_exec(db,
      "CREATE TABLE IF NOT EXISTS embed.embeddings ("
      "  imgid INTEGER PRIMARY KEY,"
      "  model_id TEXT NOT NULL,"
      "  version TEXT,"
      "  timestamp INTEGER"
      ")",
      NULL, NULL, NULL);
    if(rc != SQLITE_OK) return -1;

    char sql[256];
    snprintf(sql, sizeof(sql),
      "CREATE VIRTUAL TABLE IF NOT EXISTS embed.vec_embeddings "
      "USING vec0(imgid INTEGER PRIMARY KEY, "
      "embedding float[%d])",
      DT_AI_EMBED_DIM);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    if(rc != SQLITE_OK) return -1;

    rc = sqlite3_exec(db,
      "CREATE TABLE IF NOT EXISTS embed.tag_embeddings ("
      "  tagid INTEGER PRIMARY KEY,"
      "  name TEXT NOT NULL,"
      "  embedding BLOB NOT NULL,"
      "  embedding_sum BLOB,"
      "  source TEXT NOT NULL DEFAULT 'model',"
      "  count INTEGER DEFAULT 0,"
      "  timestamp INTEGER"
      ")",
      NULL, NULL, NULL);
    if(rc != SQLITE_OK) return -1;

    from_version = 1;
  }

  // future migrations go here:
  // if(from_version == 1) { ... from_version = 2; }

  char pragma[64];
  snprintf(pragma, sizeof(pragma),
           "PRAGMA embed.user_version = %d", EMBED_DB_VERSION);
  sqlite3_exec(db, pragma, NULL, NULL, NULL);

  return EMBED_DB_VERSION;
}

// forward declarations
static void _on_filmroll_imported(gpointer instance, uint32_t film_id,
                                  gpointer user_data);
static void _import_model_tags(void);

// --- database lifecycle ---

void dt_ai_embeddings_init(void)
{
  sqlite3 *db = dt_database_get(darktable.db);
  if(!db) return;

  // register sqlite-vec extension on the main handle
  char *errmsg = NULL;
  int rc = sqlite3_vec_init(db, &errmsg, NULL);
  if(rc != SQLITE_OK)
  {
    dt_print(DT_DEBUG_AI,
             "[embeddings] failed to init sqlite-vec: %s",
             errmsg ? errmsg : "unknown error");
    sqlite3_free(errmsg);
    return;
  }

  // attach embeddings.db in same directory as library.db
  char datadir[PATH_MAX] = {0};
  dt_loc_get_user_config_dir(datadir, sizeof(datadir));

  char dbpath[PATH_MAX] = {0};
  snprintf(dbpath, sizeof(dbpath), "%s%sembeddings.db",
           datadir, G_DIR_SEPARATOR_S);

  sqlite3_stmt *stmt = NULL;
  rc = sqlite3_prepare_v2(db,
    "ATTACH DATABASE ?1 AS embed", -1, &stmt, NULL);
  if(rc != SQLITE_OK)
  {
    dt_print(DT_DEBUG_AI,
             "[embeddings] failed to prepare ATTACH: %s",
             sqlite3_errmsg(db));
    return;
  }
  sqlite3_bind_text(stmt, 1, dbpath, -1, SQLITE_TRANSIENT);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  if(rc != SQLITE_DONE)
  {
    dt_print(DT_DEBUG_AI,
             "[embeddings] failed to attach database: %s",
             sqlite3_errmsg(db));
    return;
  }

  // check schema version and migrate if needed
  int version = 0;
  sqlite3_stmt *ver_stmt = NULL;
  if(sqlite3_prepare_v2(db, "PRAGMA embed.user_version",
                         -1, &ver_stmt, NULL) == SQLITE_OK)
  {
    if(sqlite3_step(ver_stmt) == SQLITE_ROW)
      version = sqlite3_column_int(ver_stmt, 0);
    sqlite3_finalize(ver_stmt);
  }

  const int new_version = _upgrade_embed_schema(db, version);
  if(new_version < 0)
  {
    dt_print(DT_DEBUG_AI,
             "[embeddings] schema migration failed");
    sqlite3_exec(db, "DETACH DATABASE embed",
                 NULL, NULL, NULL);
    return;
  }

  _embed_attached = TRUE;
  dt_print(DT_DEBUG_AI,
           "[embeddings] database attached: %s (v%d)",
           dbpath, new_version);

  // import model tags from tags.json into the DB
  _import_model_tags();

  // connect import signal for auto-indexing
  DT_CONTROL_SIGNAL_CONNECT(DT_SIGNAL_FILMROLLS_IMPORTED,
                             _on_filmroll_imported, NULL);
}

// signal handler: index all unindexed images in the imported film roll
static void _on_filmroll_imported(gpointer instance,
                                  uint32_t film_id,
                                  gpointer user_data)
{
  if(!_embed_attached) return;
  if(!dt_conf_get_bool("plugins/ai/index_on_import")) return;

  // collect unindexed images from this film roll
  GList *images = NULL;
  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
    "SELECT id FROM main.images WHERE film_id = ?1"
    "  AND id NOT IN (SELECT imgid FROM embed.embeddings)",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, film_id);
  while(sqlite3_step(stmt) == SQLITE_ROW)
    images = g_list_prepend(images,
               GINT_TO_POINTER(sqlite3_column_int(stmt, 0)));
  sqlite3_finalize(stmt);

  if(!images) return;

  const int n = g_list_length(images);
  dt_print(DT_DEBUG_AI,
           "[embeddings] film %d imported: %d images to index",
           film_id, n);
  dt_ai_embed_batch(images);
  g_list_free(images);
}

void dt_ai_embeddings_cleanup(void)
{
  if(!_embed_attached) return;

  DT_CONTROL_SIGNAL_DISCONNECT(_on_filmroll_imported, NULL);

  sqlite3 *db = dt_database_get(darktable.db);
  if(db)
    sqlite3_exec(db, "DETACH DATABASE embed", NULL, NULL, NULL);
  _embed_attached = FALSE;
}

// --- query ---

gboolean dt_ai_embed_has(dt_imgid_t imgid)
{
  if(!_embed_attached) return FALSE;

  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT 1 FROM embed.embeddings"
                              "  WHERE imgid = ?1",
                              -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  const gboolean found = (sqlite3_step(stmt) == SQLITE_ROW);
  sqlite3_finalize(stmt);
  return found;
}

float *dt_ai_embed_get(dt_imgid_t imgid, int *dim)
{
  if(!_embed_attached) return NULL;

  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT embedding"
                              "  FROM embed.vec_embeddings"
                              "  WHERE imgid = ?1",
                              -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);

  float *result = NULL;
  if(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const void *blob = sqlite3_column_blob(stmt, 0);
    const int bytes = sqlite3_column_bytes(stmt, 0);
    if(blob && bytes == DT_AI_EMBED_DIM * (int)sizeof(float))
    {
      result = g_malloc(bytes);
      memcpy(result, blob, bytes);
      if(dim) *dim = DT_AI_EMBED_DIM;
    }
  }
  sqlite3_finalize(stmt);
  return result;
}

// --- store ---

static gboolean _store_embedding(dt_imgid_t imgid,
                                 const float *embedding,
                                 const char *model_id,
                                 const char *version)
{
  if(!_embed_attached || !embedding) return FALSE;

  sqlite3 *db = dt_database_get(darktable.db);

  sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);

  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
                              "INSERT OR REPLACE INTO embed.embeddings"
                              "  (imgid, model_id, version, timestamp)"
                              "  VALUES (?1, ?2, ?3, ?4)",
                              -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, model_id, -1,
                             SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, version ? version : "", -1,
                             SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT64(stmt, 4, (int64_t)time(NULL));
  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  if(rc != SQLITE_DONE)
  {
    sqlite3_exec(db, "ROLLBACK", NULL, NULL, NULL);
    dt_print(DT_DEBUG_AI,
             "[embeddings] failed to store imgid %d: %s",
             imgid, sqlite3_errmsg(db));
    return FALSE;
  }

  DT_DEBUG_SQLITE3_PREPARE_V2(db,
                              "INSERT OR REPLACE INTO embed.vec_embeddings"
                              "  (imgid, embedding)"
                              "  VALUES (?1, ?2)",
                              -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 2, embedding,
                             DT_AI_EMBED_DIM * sizeof(float),
                             SQLITE_TRANSIENT);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  if(rc != SQLITE_DONE)
  {
    sqlite3_exec(db, "ROLLBACK", NULL, NULL, NULL);
    dt_print(DT_DEBUG_AI,
             "[embeddings] failed to store imgid %d: %s",
             imgid, sqlite3_errmsg(db));
    return FALSE;
  }

  sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
  return TRUE;
}

// --- compute ---

// preprocessing parameters resolved from a model's manifest attributes.
// every embedding model must declare these in its model.yaml so darktable
// doesn't bake in assumptions specific to one architecture.
typedef struct _embed_preproc_t
{
  int   input_size;        // square input edge, e.g. 224 or 384
  gboolean rgb;            // TRUE = RGB channel order, FALSE = BGR
  float input_scale;       // pre-mean multiplier, e.g. 1/255 for uint8 → [0, 1]
  float mean[3];           // per-channel mean subtracted after scaling
  float std[3];            // per-channel std dividing after mean subtraction
  gboolean output_normalized; // TRUE = model output is already L2-normalized
} _embed_preproc_t;

static void _resolve_preproc(const dt_ai_model_info_t *info,
                             _embed_preproc_t *pp)
{
  // defaults match the CLIP/ImageNet preprocessing that legacy models used
  pp->input_size = 224;
  pp->rgb = TRUE;
  pp->input_scale = 1.0f / 255.0f;
  pp->mean[0] = 0.0f; pp->mean[1] = 0.0f; pp->mean[2] = 0.0f;
  pp->std[0]  = 1.0f; pp->std[1]  = 1.0f; pp->std[2]  = 1.0f;
  pp->output_normalized = TRUE;

  if(!info) return;

  int n = 0;
  int *sizes = dt_ai_model_attribute_int_array(info, "input_sizes", &n);
  if(sizes && n > 0) pp->input_size = sizes[0];
  g_free(sizes);

  char *cs = dt_ai_model_attribute_string(info, "color_space");
  if(cs)
  {
    if(!g_ascii_strcasecmp(cs, "bgr")) pp->rgb = FALSE;
    g_free(cs);
  }

  pp->input_scale = (float)dt_ai_model_attribute_double(
    info, "input_scale", pp->input_scale);

  double *m = dt_ai_model_attribute_double_array(info, "norm_mean", &n);
  if(m && n >= 3)
  {
    pp->mean[0] = (float)m[0]; pp->mean[1] = (float)m[1]; pp->mean[2] = (float)m[2];
  }
  g_free(m);

  double *s = dt_ai_model_attribute_double_array(info, "norm_std", &n);
  if(s && n >= 3)
  {
    pp->std[0] = (float)s[0]; pp->std[1] = (float)s[1]; pp->std[2] = (float)s[2];
  }
  g_free(s);

  // explicit override only — defaults to TRUE since current models
  // (OpenCLIP) bake L2 norm into the graph
  char *norm_str = dt_ai_model_attribute_string(info, "output_l2_normalized");
  if(norm_str)
  {
    if(!g_ascii_strcasecmp(norm_str, "false")
       || !g_ascii_strcasecmp(norm_str, "no")
       || !g_ascii_strcasecmp(norm_str, "0"))
      pp->output_normalized = FALSE;
    g_free(norm_str);
  }
}

// compute embedding for one image using a pre-loaded model context.
// info carries preprocessing/output attributes from the model manifest.
static gboolean _embed_compute_with_ctx(dt_imgid_t imgid,
                                        dt_ai_context_t *ctx,
                                        const char *model_id,
                                        const dt_ai_model_info_t *info)
{
  if(!_embed_attached) return FALSE;
  if(dt_ai_embed_has(imgid)) return TRUE;

  _embed_preproc_t pp;
  _resolve_preproc(info, &pp);

  // get thumbnail from mipmap cache (DT_MIPMAP_1 = 360x225)
  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(&buf, imgid, DT_MIPMAP_1,
                      DT_MIPMAP_BLOCKING, 'r');
  if(!buf.buf || buf.width <= 0 || buf.height <= 0)
  {
    dt_mipmap_cache_release(&buf);
    return FALSE;
  }

  // BGRA 8-bit → float, bilinear resize to (input_size × input_size),
  // optional mean/std normalization per the model's preprocessing config.
  const int src_w = buf.width;
  const int src_h = buf.height;
  const int dst = pp.input_size;
  float *input = g_try_malloc(dst * dst * 3 * sizeof(float));
  if(!input)
  {
    dt_mipmap_cache_release(&buf);
    return FALSE;
  }

  for(int y = 0; y < dst; y++)
  {
    const float sy = (float)y * (src_h - 1) / (dst - 1);
    const int y0 = MIN((int)sy, src_h - 1);
    const int y1 = MIN(y0 + 1, src_h - 1);
    const float fy = sy - (float)y0;

    for(int x = 0; x < dst; x++)
    {
      const float sx = (float)x * (src_w - 1) / (dst - 1);
      const int x0 = MIN((int)sx, src_w - 1);
      const int x1 = MIN(x0 + 1, src_w - 1);
      const float fx = sx - (float)x0;

      // cairo BGRA: B=0, G=1, R=2, A=3
      const uint8_t *p00 = buf.buf + ((size_t)y0 * src_w + x0) * 4;
      const uint8_t *p01 = buf.buf + ((size_t)y0 * src_w + x1) * 4;
      const uint8_t *p10 = buf.buf + ((size_t)y1 * src_w + x0) * 4;
      const uint8_t *p11 = buf.buf + ((size_t)y1 * src_w + x1) * 4;

      for(int c = 0; c < 3; c++)
      {
        // sc maps the output channel c to the cairo BGRA byte index
        const int sc = pp.rgb ? (2 - c) : c;
        const float raw = p00[sc] * (1.f - fx) * (1.f - fy)
                        + p01[sc] * fx * (1.f - fy)
                        + p10[sc] * (1.f - fx) * fy
                        + p11[sc] * fx * fy;
        const float v = (raw * pp.input_scale - pp.mean[c]) / pp.std[c];
        input[c * dst * dst + y * dst + x] = v;
      }
    }
  }
  dt_mipmap_cache_release(&buf);

  int64_t in_shape[] = {1, 3, dst, dst};
  dt_ai_tensor_t in_tensor = {
    .data = input,
    .type = DT_AI_FLOAT,
    .shape = in_shape,
    .ndim = 4
  };

  float output[DT_AI_EMBED_DIM];
  int64_t out_shape[] = {1, DT_AI_EMBED_DIM};
  dt_ai_tensor_t out_tensor = {
    .data = output,
    .type = DT_AI_FLOAT,
    .shape = out_shape,
    .ndim = 2
  };

  const int ret = dt_ai_run(ctx, &in_tensor, 1,
                            &out_tensor, 1);
  g_free(input);

  if(ret != 0)
  {
    dt_print(DT_DEBUG_AI,
             "[embeddings] inference failed for imgid %d", imgid);
    return FALSE;
  }

  // models that don't bake L2 norm into the graph get it here, so
  // every embedding stored in the DB is unit-length and the cosine-
  // similarity tag matching can rely on a plain dot product
  if(!pp.output_normalized)
    _normalize(output, DT_AI_EMBED_DIM);

  const gboolean ok = _store_embedding(imgid, output,
                                       model_id, NULL);
  if(ok)
    dt_print(DT_DEBUG_AI,
             "[embeddings] indexed imgid %d", imgid);

  return ok;
}

gboolean dt_ai_embed_compute(dt_imgid_t imgid)
{
  if(!_embed_attached) return FALSE;

  char *model_id = dt_ai_models_get_active_for_task(EMBED_MODEL_TASK);
  if(!model_id || !model_id[0])
  {
    g_free(model_id);
    return FALSE;
  }

  dt_ai_environment_t *env = dt_ai_env_init(NULL);
  if(!env) { g_free(model_id); return FALSE; }

  dt_ai_context_t *ctx
    = dt_ai_load_model(env, model_id, NULL, DT_AI_PROVIDER_AUTO);
  if(!ctx)
  {
    dt_ai_env_destroy(env);
    g_free(model_id);
    return FALSE;
  }

  const dt_ai_model_info_t *info
    = dt_ai_get_model_info_by_id(env, model_id);
  const gboolean ok = _embed_compute_with_ctx(imgid, ctx, model_id, info);

  dt_ai_unload_model(ctx);
  dt_ai_env_destroy(env);
  g_free(model_id);
  return ok;
}

// --- batch job ---

typedef struct _embed_job_t
{
  GList *images;
} _embed_job_t;

static int32_t _embed_job_run(dt_job_t *job)
{
  _embed_job_t *j = dt_control_job_get_params(job);
  if(!j) return 1;

  char *model_id = dt_ai_models_get_active_for_task(EMBED_MODEL_TASK);
  if(!model_id || !model_id[0])
  {
    g_free(model_id);
    dt_print(DT_DEBUG_ALWAYS,
             "[embeddings] indexing skipped: no embed model enabled. "
             "download from preferences → AI");
    return 1;
  }

  dt_ai_environment_t *env = dt_ai_env_init(NULL);
  if(!env) { g_free(model_id); return 1; }

  dt_ai_context_t *ctx
    = dt_ai_load_model(env, model_id, NULL, DT_AI_PROVIDER_AUTO);
  if(!ctx)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[embeddings] indexing skipped: failed to load model %s. "
             "download from preferences → AI",
             model_id);
    dt_ai_env_destroy(env);
    g_free(model_id);
    return 1;
  }

  const dt_ai_model_info_t *info
    = dt_ai_get_model_info_by_id(env, model_id);

  const int total = g_list_length(j->images);
  int done = 0;
  const gboolean auto_tag
    = dt_conf_get_bool("plugins/ai/auto_tag");

  // refresh user tag centroids before auto-tagging
  if(auto_tag)
    dt_ai_embed_update_user_tags();

  dt_control_job_set_progress_message(job,
    ngettext("indexing %d image", "indexing %d images", total),
    total);

  for(GList *l = j->images; l; l = g_list_next(l))
  {
    if(dt_control_job_get_state(job) == DT_JOB_STATE_CANCELLED)
      break;

    const dt_imgid_t imgid = GPOINTER_TO_INT(l->data);
    if(_embed_compute_with_ctx(imgid, ctx, model_id, info) && auto_tag)
      dt_ai_embed_auto_tag(imgid);

    done++;
    dt_control_job_set_progress(job, (double)done / total);
    dt_control_job_set_progress_message(job,
      _("indexing %d/%d images"), done, total);

  }

  dt_ai_unload_model(ctx);
  dt_ai_env_destroy(env);
  g_free(model_id);

  dt_print(DT_DEBUG_AI,
           "[embeddings] batch complete: %d/%d indexed",
           done, total);
  return 0;
}

static void _embed_job_cleanup(void *param)
{
  _embed_job_t *j = param;
  g_list_free(j->images);
  g_free(j);
}

// --- tag embeddings ---

#define TAG_SIMILARITY_THRESHOLD 0.2f

// import pre-computed tag embeddings from the model's tags.json
// into embed.tag_embeddings with source='model'
static void _import_model_tags(void)
{
  if(!_embed_attached) return;

  char *model_id
    = dt_ai_models_get_active_for_task(EMBED_MODEL_TASK);
  if(!model_id) return;

  char *model_path = dt_ai_models_get_path(model_id);
  g_free(model_id);
  if(!model_path) return;

  char *json_path
    = g_build_filename(model_path, "tags.json", NULL);
  g_free(model_path);

  JsonParser *parser = json_parser_new();
  if(!json_parser_load_from_file(parser, json_path, NULL))
  {
    g_free(json_path);
    g_object_unref(parser);
    return;
  }
  g_free(json_path);

  JsonNode *root = json_parser_get_root(parser);
  JsonObject *obj = json_node_get_object(root);
  JsonArray *tags = json_object_get_array_member(obj, "tags");
  JsonArray *embeds
    = json_object_get_array_member(obj, "embeddings");

  if(!tags || !embeds)
  {
    g_object_unref(parser);
    return;
  }

  const int n = (int)json_array_get_length(tags);
  const int ne = (int)json_array_get_length(embeds);
  if(n != ne || n <= 0)
  {
    g_object_unref(parser);
    return;
  }

  sqlite3 *db = dt_database_get(darktable.db);
  int imported = 0;

  for(int i = 0; i < n; i++)
  {
    const char *tag_name
      = json_array_get_string_element(tags, i);
    JsonArray *vec = json_array_get_array_element(embeds, i);
    const int vlen = MIN((int)json_array_get_length(vec),
                         DT_AI_EMBED_DIM);

    // create tag in darktable's dictionary
    guint tagid = 0;
    dt_tag_new(tag_name, &tagid);
    if(tagid == 0) continue;

    // parse embedding vector
    float embedding[DT_AI_EMBED_DIM];
    memset(embedding, 0, sizeof(embedding));
    for(int d = 0; d < vlen; d++)
      embedding[d]
        = (float)json_array_get_double_element(vec, d);

    // store — model tags don't overwrite user tags
    sqlite3_stmt *stmt = NULL;
    DT_DEBUG_SQLITE3_PREPARE_V2(db,
      "INSERT OR IGNORE INTO embed.tag_embeddings"
      "  (tagid, name, embedding, source, count, timestamp)"
      "  VALUES (?1, ?2, ?3, 'model', 0, ?4)",
      -1, &stmt, NULL);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, tagid);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, tag_name, -1,
                               SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 3, embedding,
                               DT_AI_EMBED_DIM * sizeof(float),
                               SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_INT64(stmt, 4, (int64_t)time(NULL));
    if(sqlite3_step(stmt) == SQLITE_DONE) imported++;
    sqlite3_finalize(stmt);
  }

  g_object_unref(parser);

  dt_print(DT_DEBUG_AI,
           "[embeddings] imported %d/%d model tags", imported, n);
}

// normalize a float vector to unit length in-place
static void _normalize(float *vec, int dim)
{
  double norm = 0.0;
  for(int d = 0; d < dim; d++)
    norm += (double)vec[d] * (double)vec[d];
  norm = sqrt(norm);
  if(norm < 1e-8) return;
  for(int d = 0; d < dim; d++)
    vec[d] = (float)((double)vec[d] / norm);
}

// store updated sum, count, and normalized embedding for a user tag
static void _store_user_tag(sqlite3 *db, int tagid,
                            const char *name,
                            const float *sum,
                            int count)
{
  float embedding[DT_AI_EMBED_DIM];
  memcpy(embedding, sum, sizeof(embedding));
  _normalize(embedding, DT_AI_EMBED_DIM);

  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    "INSERT OR REPLACE INTO embed.tag_embeddings"
    "  (tagid, name, embedding, embedding_sum,"
    "   source, count, timestamp)"
    "  VALUES (?1, ?2, ?3, ?4, 'user', ?5, ?6)",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, tagid);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, name, -1,
                             SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 3, embedding,
                             DT_AI_EMBED_DIM * sizeof(float),
                             SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 4, sum,
                             DT_AI_EMBED_DIM * sizeof(float),
                             SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 5, count);
  DT_DEBUG_SQLITE3_BIND_INT64(stmt, 6, (int64_t)time(NULL));
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

// full recompute of all user tag centroids from scratch.
// used before auto-tagging and after model changes
void dt_ai_embed_update_user_tags(void)
{
  if(!_embed_attached) return;

  sqlite3 *db = dt_database_get(darktable.db);

  // find tags with enough indexed images
  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    "SELECT ti.tagid, t.name, COUNT(*) AS cnt"
    "  FROM main.tagged_images ti"
    "  JOIN data.tags t ON t.id = ti.tagid"
    "  JOIN embed.embeddings e ON e.imgid = ti.imgid"
    "  GROUP BY ti.tagid"
    "  HAVING cnt >= ?1",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, TAG_MIN_EXAMPLES);

  int updated = 0;

  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const int tagid = sqlite3_column_int(stmt, 0);
    const char *name
      = (const char *)sqlite3_column_text(stmt, 1);

    // compute sum of all image embeddings with this tag
    float sum[DT_AI_EMBED_DIM];
    memset(sum, 0, sizeof(sum));

    sqlite3_stmt *vec_stmt = NULL;
    DT_DEBUG_SQLITE3_PREPARE_V2(db,
      "SELECT v.embedding FROM embed.vec_embeddings v"
      "  JOIN main.tagged_images ti"
      "    ON ti.imgid = v.imgid"
      "  WHERE ti.tagid = ?1",
      -1, &vec_stmt, NULL);
    DT_DEBUG_SQLITE3_BIND_INT(vec_stmt, 1, tagid);

    int n = 0;
    while(sqlite3_step(vec_stmt) == SQLITE_ROW)
    {
      const float *vec = sqlite3_column_blob(vec_stmt, 0);
      const int bytes = sqlite3_column_bytes(vec_stmt, 0);
      if(!vec
         || bytes != DT_AI_EMBED_DIM * (int)sizeof(float))
        continue;
      for(int d = 0; d < DT_AI_EMBED_DIM; d++)
        sum[d] += vec[d];
      n++;
    }
    sqlite3_finalize(vec_stmt);

    if(n < TAG_MIN_EXAMPLES) continue;

    _store_user_tag(db, tagid, name, sum, n);
    updated++;
  }
  sqlite3_finalize(stmt);

  if(updated > 0)
    dt_print(DT_DEBUG_AI,
             "[embeddings] updated %d user tag embeddings",
             updated);
}

void dt_ai_embed_auto_tag(dt_imgid_t imgid)
{
  if(!_embed_attached) return;

  int dim = 0;
  float *embedding = dt_ai_embed_get(imgid, &dim);
  if(!embedding || dim != DT_AI_EMBED_DIM) return;

  const float threshold
    = dt_conf_key_exists("plugins/ai/auto_tag_threshold")
      ? dt_conf_get_float("plugins/ai/auto_tag_threshold")
      : TAG_SIMILARITY_THRESHOLD;

  sqlite3 *db = dt_database_get(darktable.db);
  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    "SELECT tagid, embedding"
    "  FROM embed.tag_embeddings",
    -1, &stmt, NULL);

  int applied = 0;
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const int tagid = sqlite3_column_int(stmt, 0);
    const float *tag_vec = sqlite3_column_blob(stmt, 1);
    const int bytes = sqlite3_column_bytes(stmt, 1);

    if(!tag_vec
       || bytes != DT_AI_EMBED_DIM * (int)sizeof(float))
      continue;

    // cosine similarity (both vectors are L2-normalized)
    double dot = 0.0;
    for(int d = 0; d < DT_AI_EMBED_DIM; d++)
      dot += (double)embedding[d] * (double)tag_vec[d];

    if((float)dot >= threshold)
    {
      dt_tag_attach(tagid, imgid, FALSE, FALSE);
      applied++;
    }
  }
  sqlite3_finalize(stmt);
  g_free(embedding);

  if(applied > 0)
    dt_print(DT_DEBUG_AI,
             "[embeddings] auto-tagged imgid %d: %d tags",
             imgid, applied);
}

void dt_ai_embed_batch(GList *images)
{
  if(!images || !_embed_attached) return;

  _embed_job_t *j = g_new0(_embed_job_t, 1);
  j->images = g_list_copy(images);

  dt_job_t *job = dt_control_job_create(_embed_job_run,
                                        "ai embed");
  dt_control_job_set_params(job, j, _embed_job_cleanup);
  dt_control_job_add_progress(job, _("indexing images"), TRUE);
  dt_control_add_job(DT_JOB_QUEUE_USER_BG, job);
}

void dt_ai_embed_remove(GList *images)
{
  if(!images || !_embed_attached) return;

  sqlite3 *db = dt_database_get(darktable.db);
  sqlite3_stmt *stmt_meta = NULL, *stmt_vec = NULL;

  DT_DEBUG_SQLITE3_PREPARE_V2(db,
                              "DELETE FROM embed.embeddings WHERE imgid = ?1",
                              -1, &stmt_meta, NULL);
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
                              "DELETE FROM embed.vec_embeddings WHERE imgid = ?1",
                              -1, &stmt_vec, NULL);

  DT_DEBUG_SQLITE3_EXEC(db, "BEGIN TRANSACTION", NULL, NULL, NULL);
  int removed = 0;
  for(GList *l = images; l; l = g_list_next(l))
  {
    const dt_imgid_t imgid = GPOINTER_TO_INT(l->data);
    sqlite3_bind_int(stmt_meta, 1, imgid);
    sqlite3_bind_int(stmt_vec, 1, imgid);
    if(sqlite3_step(stmt_meta) == SQLITE_DONE
       && sqlite3_step(stmt_vec) == SQLITE_DONE)
      removed++;
    sqlite3_reset(stmt_meta);
    sqlite3_reset(stmt_vec);
    sqlite3_clear_bindings(stmt_meta);
    sqlite3_clear_bindings(stmt_vec);
  }
  DT_DEBUG_SQLITE3_EXEC(db, "COMMIT", NULL, NULL, NULL);

  sqlite3_finalize(stmt_meta);
  sqlite3_finalize(stmt_vec);

  dt_print(DT_DEBUG_AI,
           "[embeddings] removed %d image(s) from index", removed);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
