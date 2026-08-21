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

#define EMBED_MODEL_TASK "embedding"
#define EMBED_DB_VERSION 1
#define TAG_MIN_EXAMPLES 3

// embeddings DB is attached to the main database handle as "embed"
static gboolean _embed_attached = FALSE;

// forward decl: definition lives next to the tag-centroid helpers below
static void _normalize(float *vec, int dim);

// upgrade the embeddings database schema step by step.
// returns the new version, or -1 on error
// records which (image, tag) pairs darktable applied itself. centroids
// and the LOO estimate must be built from human-applied tags only, or
// the model trains on its own output and the gate ratchets itself open
static int _create_auto_tag_table(sqlite3 *db)
{
  return sqlite3_exec(db,
    "CREATE TABLE IF NOT EXISTS embed.auto_tagged ("
    "  imgid INTEGER NOT NULL,"
    "  tagid INTEGER NOT NULL,"
    "  PRIMARY KEY (imgid, tagid)"
    ")",
    NULL, NULL, NULL);
}

static int _create_tag_table(sqlite3 *db)
{
  return sqlite3_exec(db,
    "CREATE TABLE IF NOT EXISTS embed.tag_embeddings ("
    "  tagid INTEGER PRIMARY KEY,"
    "  name TEXT NOT NULL,"
    "  embedding BLOB NOT NULL,"
    "  embedding_sum BLOB,"
    "  source TEXT NOT NULL DEFAULT 'model',"
    "  count INTEGER DEFAULT 0,"
    "  score_mean REAL,"
    "  score_std REAL,"
    "  coherence REAL,"
    "  separation REAL,"
    "  loo_recall REAL,"
    "  timestamp INTEGER"
    ")",
    NULL, NULL, NULL);
}

// DEVELOPMENT ONLY - remove when the schema freezes at release.
// the schema version stays at 1 while the shape is still moving, so a
// changed tag_embeddings is not caught by the version check and the
// first query for a new column fails instead. tag_embeddings is purely
// derived (model centroids from tags.json, user centroids from tagged
// images), so rebuilding it costs nothing and the next indexing batch
// repopulates it. image embeddings are never touched - those cost an
// inference run each
static void _dev_ensure_tag_table_shape(sqlite3 *db)
{
  gboolean has_scores = FALSE;
  sqlite3_stmt *stmt = NULL;
  if(sqlite3_prepare_v2(db, "PRAGMA embed.table_info(tag_embeddings)",
                        -1, &stmt, NULL) == SQLITE_OK)
  {
    while(sqlite3_step(stmt) == SQLITE_ROW)
    {
      const char *col = (const char *)sqlite3_column_text(stmt, 1);
      if(!g_strcmp0(col, "loo_recall")) has_scores = TRUE;
    }
    sqlite3_finalize(stmt);
  }
  if(has_scores) return;

  dt_print(DT_DEBUG_AI,
           "[embeddings] tag table shape is stale, rebuilding");
  sqlite3_exec(db, "DROP TABLE IF EXISTS embed.tag_embeddings",
               NULL, NULL, NULL);
  _create_tag_table(db);
  _create_auto_tag_table(db);
}

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

    if(_create_tag_table(db) != SQLITE_OK) return -1;
    if(_create_auto_tag_table(db) != SQLITE_OK) return -1;

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
static void _evaluate_user_tags(void);

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

  // see the comment on this function: development-only, drop at
  // release. must come after the failure check - a database from a
  // newer darktable is refused above, and dropping its tag table
  // would damage a library this build just declined to open
  _dev_ensure_tag_table_shape(db);

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

gboolean dt_ai_embed_has_any(GList *images)
{
  if(!_embed_attached || !images) return FALSE;

  GString *ids = g_string_new(NULL);
  for(GList *l = images; l; l = g_list_next(l))
    g_string_append_printf(ids, "%s%d", ids->len ? "," : "",
                           GPOINTER_TO_INT(l->data));

  char *q = g_strdup_printf(
    "SELECT 1 FROM embed.embeddings WHERE imgid IN (%s) LIMIT 1",
    ids->str);
  g_string_free(ids, TRUE);

  sqlite3_stmt *stmt = NULL;
  gboolean found = FALSE;
  if(sqlite3_prepare_v2(dt_database_get(darktable.db), q, -1,
                        &stmt, NULL) == SQLITE_OK)
  {
    found = (sqlite3_step(stmt) == SQLITE_ROW);
    sqlite3_finalize(stmt);
  }
  g_free(q);
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

  // darktable nests transactions through a refcount; issuing raw
  // BEGIN/COMMIT on the shared handle from the worker thread can
  // commit or roll back whatever the main thread has open
  dt_database_start_transaction(darktable.db);

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
    dt_database_rollback_transaction(darktable.db);
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
    dt_database_rollback_transaction(darktable.db);
    dt_print(DT_DEBUG_AI,
             "[embeddings] failed to store imgid %d: %s",
             imgid, sqlite3_errmsg(db));
    return FALSE;
  }

  dt_database_release_transaction(darktable.db);
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

  // CLIP preprocessing is "resize shortest side, then centre crop" —
  // never anamorphic. sampling the largest centred square and scaling
  // it to dst x dst is the same thing in one pass. squashing instead
  // compresses every non-square photo along its long axis before the
  // encoder sees it, which degrades every embedding
  const int side = MIN(src_w, src_h);
  const int off_x = (src_w - side) / 2;
  const int off_y = (src_h - side) / 2;

  for(int y = 0; y < dst; y++)
  {
    const float sy = off_y + (float)y * (side - 1) / (dst - 1);
    const int y0 = MIN((int)sy, src_h - 1);
    const int y1 = MIN(y0 + 1, src_h - 1);
    const float fy = sy - (float)y0;

    for(int x = 0; x < dst; x++)
    {
      const float sx = off_x + (float)x * (side - 1) / (dst - 1);
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

  // dt_ai_run overwrites out_shape with the runtime's real dims for a
  // dynamic-shape model and copies that many floats, so a model with a
  // different embedding width would run off this buffer
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

  if(ret == 0 && (out_tensor.ndim != 2
                  || out_shape[1] != DT_AI_EMBED_DIM))
  {
    dt_print(DT_DEBUG_AI,
             "[embeddings] model returned %" PRId64 " dims, expected %d - "
             "this model is not usable for embeddings",
             out_tensor.ndim == 2 ? out_shape[1] : -1,
             DT_AI_EMBED_DIM);
    return FALSE;
  }

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

  dt_control_job_set_progress_message(job,
    ngettext("indexing %d image", "indexing %d images", total),
    total);

  GList *indexed = NULL;
  for(GList *l = j->images; l; l = g_list_next(l))
  {
    if(dt_control_job_get_state(job) == DT_JOB_STATE_CANCELLED)
      break;

    const dt_imgid_t imgid = GPOINTER_TO_INT(l->data);
    if(_embed_compute_with_ctx(imgid, ctx, model_id, info) && auto_tag)
      indexed = g_list_prepend(indexed, GINT_TO_POINTER(imgid));

    done++;
    dt_control_job_set_progress(job, (double)done / total);
    dt_control_job_set_progress_message(job,
      _("indexing %d/%d images"), done, total);

  }

  // tag only after calibrating, and calibrate only once the batch's
  // images are in the index: the statistics describe the distribution
  // these images are part of. tagging before this point would score
  // every tag against nothing and emit top-K per group unfiltered
  const gboolean cancelled
    = dt_control_job_get_state(job) == DT_JOB_STATE_CANCELLED;

  if(auto_tag && indexed && !cancelled)
  {
    // every quality measure is relative to the library distribution,
    // so the order is fixed: centroids, then library statistics, then
    // per-tag quality, then apply
    dt_ai_embed_update_user_tags();
    dt_ai_embed_calibrate_tags();
    _evaluate_user_tags();
    for(GList *l = indexed; l; l = g_list_next(l))
      dt_ai_embed_auto_tag(GPOINTER_TO_INT(l->data));
  }
  g_list_free(indexed);

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

// z-score floor: how many standard deviations closer than typical a
// tag must be before it is applied at all
#define TAG_ZSCORE_FLOOR 1.5f
// how many tags to apply per taxonomy group ("genre", "subject", ...)
#define TAG_TOP_K_PER_GROUP 2
// images sampled when calibrating per-tag score statistics
#define TAG_CALIBRATION_SAMPLE 500
// a prototype built from a handful of images mostly encodes what those
// images incidentally share - location, light, day - rather than the
// concept. below this the LOO estimate is variance, not signal
#define TAG_MIN_EXAMPLES_APPLY 15
// leave-one-out recall a tag must reach before it is auto-applied
#define TAG_MIN_LOO_RECALL 0.6f

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
  char *model_id = dt_ai_models_get_active_for_task(EMBED_MODEL_TASK);
  if(!model_id || !model_id[0]) { g_free(model_id); return; }

  // find tags with enough indexed images
  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    "SELECT ti.tagid, t.name, COUNT(*) AS cnt"
    "  FROM main.tagged_images ti"
    "  JOIN data.tags t ON t.id = ti.tagid"
    "  JOIN embed.embeddings e ON e.imgid = ti.imgid"
    " WHERE t.name NOT LIKE 'darktable|%'"
    "   AND e.model_id = ?2"
    "   AND NOT EXISTS (SELECT 1 FROM embed.auto_tagged a"
    "                    WHERE a.imgid = ti.imgid AND a.tagid = ti.tagid)"
    "  GROUP BY ti.tagid"
    "  HAVING cnt >= ?1",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, TAG_MIN_EXAMPLES);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, model_id, -1, SQLITE_TRANSIENT);

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
      "  JOIN main.tagged_images ti ON ti.imgid = v.imgid"
      "  JOIN embed.embeddings e ON e.imgid = v.imgid"
      " WHERE ti.tagid = ?1"
      "   AND e.model_id = ?2"
      "   AND NOT EXISTS (SELECT 1 FROM embed.auto_tagged a"
      "                    WHERE a.imgid = v.imgid AND a.tagid = ti.tagid)",
      -1, &vec_stmt, NULL);
    DT_DEBUG_SQLITE3_BIND_INT(vec_stmt, 1, tagid);
    DT_DEBUG_SQLITE3_BIND_TEXT(vec_stmt, 2, model_id, -1, SQLITE_TRANSIENT);

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
  g_free(model_id);
}

// measure each tag's cosine distribution over a sample of the library.
// this is what makes scores comparable: a tag that sits near every
// image gets a high mean, so it only wins when an image is unusually
// close *for that tag*
// per-tag quality, evaluated after calibration because every measure
// is expressed relative to the library-wide distribution.
//
//   coherence  mean cosine of a tag's examples to their own prototype
//   separation that coherence as a z-score against the library. a tag
//              covering most of someone's library is compact AND
//              worthless, so compactness alone cannot be the gate
//   loo_recall leave-one-out estimate of how often a *new* image of
//              this kind would be recognised. this is PU data - the
//              untagged remainder contains unlabelled positives - so
//              only recall is estimable here, not precision
//
// the LOO fold is O(d): the stored sum is a sufficient statistic, so
// dropping one example is a subtraction rather than a rebuild
static void _evaluate_user_tags(void)
{
  if(!_embed_attached) return;

  sqlite3 *db = dt_database_get(darktable.db);
  char *model_id = dt_ai_models_get_active_for_task(EMBED_MODEL_TASK);
  if(!model_id || !model_id[0]) { g_free(model_id); return; }
  const float gate_z
    = dt_conf_key_exists("plugins/ai/auto_tag_zscore")
      ? dt_conf_get_float("plugins/ai/auto_tag_zscore")
      : TAG_ZSCORE_FLOOR;

  sqlite3_stmt *tags = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    "SELECT tagid, embedding_sum, count, score_mean, score_std"
    "  FROM embed.tag_embeddings"
    " WHERE source = 'user' AND score_mean IS NOT NULL",
    -1, &tags, NULL);

  while(sqlite3_step(tags) == SQLITE_ROW)
  {
    const int tagid = sqlite3_column_int(tags, 0);
    const float *sum = sqlite3_column_blob(tags, 1);
    const int n = sqlite3_column_int(tags, 2);
    const double mean = sqlite3_column_double(tags, 3);
    const double sd = MAX(sqlite3_column_double(tags, 4), 1e-6);

    if(!sum || n < 2
       || sqlite3_column_bytes(tags, 1)
            != DT_AI_EMBED_DIM * (int)sizeof(float))
      continue;

    // NB: positives must be human-applied only. auto-applied tags
    // feeding back in would make this estimate self-fulfilling
    sqlite3_stmt *ex = NULL;
    DT_DEBUG_SQLITE3_PREPARE_V2(db,
      "SELECT v.embedding FROM embed.vec_embeddings v"
      "  JOIN main.tagged_images ti ON ti.imgid = v.imgid"
      "  JOIN embed.embeddings e ON e.imgid = v.imgid"
      " WHERE ti.tagid = ?1"
      "   AND e.model_id = ?2"
      "   AND NOT EXISTS (SELECT 1 FROM embed.auto_tagged a"
      "                    WHERE a.imgid = v.imgid AND a.tagid = ti.tagid)",
      -1, &ex, NULL);
    DT_DEBUG_SQLITE3_BIND_INT(ex, 1, tagid);
    DT_DEBUG_SQLITE3_BIND_TEXT(ex, 2, model_id, -1, SQLITE_TRANSIENT);

    float full[DT_AI_EMBED_DIM];
    memcpy(full, sum, sizeof(full));
    _normalize(full, DT_AI_EMBED_DIM);

    double coh_sum = 0.0;
    int seen = 0, recalled = 0;

    while(sqlite3_step(ex) == SQLITE_ROW)
    {
      const float *x = sqlite3_column_blob(ex, 0);
      if(!x || sqlite3_column_bytes(ex, 0)
                 != DT_AI_EMBED_DIM * (int)sizeof(float))
        continue;

      double dot_full = 0.0;
      float loo[DT_AI_EMBED_DIM];
      for(int d = 0; d < DT_AI_EMBED_DIM; d++)
      {
        dot_full += (double)x[d] * (double)full[d];
        loo[d] = (sum[d] - x[d]) / (float)(n - 1);
      }
      coh_sum += dot_full;

      _normalize(loo, DT_AI_EMBED_DIM);
      double dot_loo = 0.0;
      for(int d = 0; d < DT_AI_EMBED_DIM; d++)
        dot_loo += (double)x[d] * (double)loo[d];

      if((dot_loo - mean) / sd >= gate_z) recalled++;
      seen++;
    }
    sqlite3_finalize(ex);

    if(seen < 2) continue;

    const double coherence = coh_sum / seen;
    const double separation = (coherence - mean) / sd;
    const double loo_recall = (double)recalled / seen;

    sqlite3_stmt *upd = NULL;
    DT_DEBUG_SQLITE3_PREPARE_V2(db,
      "UPDATE embed.tag_embeddings"
      "   SET coherence = ?2, separation = ?3, loo_recall = ?4"
      " WHERE tagid = ?1",
      -1, &upd, NULL);
    DT_DEBUG_SQLITE3_BIND_INT(upd, 1, tagid);
    sqlite3_bind_double(upd, 2, coherence);
    sqlite3_bind_double(upd, 3, separation);
    sqlite3_bind_double(upd, 4, loo_recall);
    sqlite3_step(upd);
    sqlite3_finalize(upd);

    dt_print(DT_DEBUG_AI,
             "[embeddings] tag %d: n=%d coherence=%.3f "
             "separation=%.2f loo_recall=%.2f%s",
             tagid, seen, coherence, separation, loo_recall,
             (seen >= TAG_MIN_EXAMPLES_APPLY
              && loo_recall >= TAG_MIN_LOO_RECALL) ? "" : "  (below gate)");
  }
  sqlite3_finalize(tags);
  g_free(model_id);
}

void dt_ai_embed_calibrate_tags(void)
{
  if(!_embed_attached) return;

  sqlite3 *db = dt_database_get(darktable.db);

  // sample of image embeddings
  GPtrArray *sample = g_ptr_array_new_with_free_func(g_free);
  sqlite3_stmt *stmt = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    // random, not the first N: LIMIT without ORDER BY returns the
    // oldest-indexed images, which biases every tag's mean and std
    "SELECT embedding FROM embed.vec_embeddings"
    "  ORDER BY random() LIMIT ?1",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, TAG_CALIBRATION_SAMPLE);
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const float *v = sqlite3_column_blob(stmt, 0);
    if(v && sqlite3_column_bytes(stmt, 0)
             == DT_AI_EMBED_DIM * (int)sizeof(float))
    {
      float *copy = g_malloc(DT_AI_EMBED_DIM * sizeof(float));
      memcpy(copy, v, DT_AI_EMBED_DIM * sizeof(float));
      g_ptr_array_add(sample, copy);
    }
  }
  sqlite3_finalize(stmt);

  // with too few images the statistics are noise; leave the columns
  // NULL and let matching fall back to raw cosine
  if(sample->len < 16)
  {
    g_ptr_array_free(sample, TRUE);
    return;
  }

  GArray *ids = g_array_new(FALSE, FALSE, sizeof(int));
  GPtrArray *vecs = g_ptr_array_new_with_free_func(g_free);
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    "SELECT tagid, embedding FROM embed.tag_embeddings",
    -1, &stmt, NULL);
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const float *v = sqlite3_column_blob(stmt, 1);
    if(!v || sqlite3_column_bytes(stmt, 1)
               != DT_AI_EMBED_DIM * (int)sizeof(float))
      continue;
    const int tagid = sqlite3_column_int(stmt, 0);
    g_array_append_val(ids, tagid);
    float *copy = g_malloc(DT_AI_EMBED_DIM * sizeof(float));
    memcpy(copy, v, DT_AI_EMBED_DIM * sizeof(float));
    g_ptr_array_add(vecs, copy);
  }
  sqlite3_finalize(stmt);

  sqlite3_stmt *upd = NULL;
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    "UPDATE embed.tag_embeddings"
    "   SET score_mean = ?2, score_std = ?3"
    " WHERE tagid = ?1",
    -1, &upd, NULL);

  for(guint t = 0; t < vecs->len; t++)
  {
    const float *tag_vec = g_ptr_array_index(vecs, t);
    double sum = 0.0, sum_sq = 0.0;

    for(guint i = 0; i < sample->len; i++)
    {
      const float *img = g_ptr_array_index(sample, i);
      double dot = 0.0;
      for(int d = 0; d < DT_AI_EMBED_DIM; d++)
        dot += (double)img[d] * (double)tag_vec[d];
      sum += dot;
      sum_sq += dot * dot;
    }

    const double n = (double)sample->len;
    const double mean = sum / n;
    const double var = MAX(sum_sq / n - mean * mean, 0.0);
    const double sd = MAX(sqrt(var), 1e-6);

    sqlite3_reset(upd);
    DT_DEBUG_SQLITE3_BIND_INT(upd, 1, g_array_index(ids, int, t));
    sqlite3_bind_double(upd, 2, mean);
    sqlite3_bind_double(upd, 3, sd);
    sqlite3_step(upd);
  }
  sqlite3_finalize(upd);

  dt_print(DT_DEBUG_AI,
           "[embeddings] calibrated %u tags over %u images",
           vecs->len, sample->len);

  g_array_free(ids, TRUE);
  g_ptr_array_free(vecs, TRUE);
  g_ptr_array_free(sample, TRUE);
}

// first path component of a hierarchical tag: "subject|animal|dog"
// -> "subject". tags compete only within their own group, because the
// taxonomy's groups are not mutually exclusive - an image has a genre
// *and* a subject *and* a time of day
static char *_tag_group(const char *name)
{
  const char *bar = strchr(name, '|');
  return bar ? g_strndup(name, bar - name) : g_strdup(name);
}

typedef struct _tag_hit_t
{
  int tagid;
  float score;
} _tag_hit_t;

static gint _hit_cmp(gconstpointer a, gconstpointer b)
{
  const float d = ((const _tag_hit_t *)b)->score
                - ((const _tag_hit_t *)a)->score;
  return (d > 0) - (d < 0);
}

void dt_ai_embed_auto_tag(dt_imgid_t imgid)
{
  if(!_embed_attached) return;

  int dim = 0;
  float *embedding = dt_ai_embed_get(imgid, &dim);
  if(!embedding || dim != DT_AI_EMBED_DIM)
  {
    g_free(embedding);
    return;
  }

  sqlite3 *db = dt_database_get(darktable.db);

  // an embedding computed by a different model is not comparable with
  // these centroids; silently scoring it would produce plausible
  // nonsense rather than an error
  char *active = dt_ai_models_get_active_for_task(EMBED_MODEL_TASK);
  sqlite3_stmt *chk = NULL;
  gboolean same_model = FALSE;
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    "SELECT model_id FROM embed.embeddings WHERE imgid = ?1",
    -1, &chk, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(chk, 1, imgid);
  if(sqlite3_step(chk) == SQLITE_ROW)
  {
    const char *stored = (const char *)sqlite3_column_text(chk, 0);
    same_model = active && stored && !strcmp(active, stored);
  }
  sqlite3_finalize(chk);
  g_free(active);

  if(!same_model)
  {
    dt_print(DT_DEBUG_AI,
             "[embeddings] imgid %d indexed by another model, skipping",
             imgid);
    g_free(embedding);
    return;
  }

  const float floor_z
    = dt_conf_key_exists("plugins/ai/auto_tag_zscore")
      ? dt_conf_get_float("plugins/ai/auto_tag_zscore")
      : TAG_ZSCORE_FLOOR;

  // group name -> GArray of _tag_hit_t
  GHashTable *groups = g_hash_table_new_full(g_str_hash, g_str_equal,
                                             g_free,
                                             (GDestroyNotify)g_array_unref);

  sqlite3_stmt *stmt = NULL;
  // only user-defined tags are auto-applied, and only those whose
  // leave-one-out estimate says they predict unseen images. the model
  // taxonomy is image-to-text, far weaker signal, and its concepts are
  // generic - it belongs in suggestions, not written to the library
  DT_DEBUG_SQLITE3_PREPARE_V2(db,
    "SELECT tagid, name, embedding, score_mean, score_std"
    "  FROM embed.tag_embeddings"
    " WHERE source = 'user'"
    "   AND count >= ?1"
    "   AND loo_recall >= ?2",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, TAG_MIN_EXAMPLES_APPLY);
  sqlite3_bind_double(stmt, 2, TAG_MIN_LOO_RECALL);

  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const int tagid = sqlite3_column_int(stmt, 0);
    const char *name = (const char *)sqlite3_column_text(stmt, 1);
    const float *tag_vec = sqlite3_column_blob(stmt, 2);
    const int bytes = sqlite3_column_bytes(stmt, 2);

    if(!name || !tag_vec
       || bytes != DT_AI_EMBED_DIM * (int)sizeof(float))
      continue;

    double dot = 0.0;
    for(int d = 0; d < DT_AI_EMBED_DIM; d++)
      dot += (double)embedding[d] * (double)tag_vec[d];

    // an uncalibrated tag is skipped, never applied on raw cosine:
    // without per-tag statistics there is nothing to compare against,
    // and top-K per group would then emit its full quota for every
    // image regardless of how poorly it matches
    if(sqlite3_column_type(stmt, 3) == SQLITE_NULL
       || sqlite3_column_type(stmt, 4) == SQLITE_NULL)
      continue;

    const double mean = sqlite3_column_double(stmt, 3);
    const double sd = MAX(sqlite3_column_double(stmt, 4), 1e-6);
    const float score = (float)((dot - mean) / sd);
    if(score < floor_z) continue;

    char *group = _tag_group(name);
    GArray *hits = g_hash_table_lookup(groups, group);
    if(!hits)
    {
      hits = g_array_new(FALSE, FALSE, sizeof(_tag_hit_t));
      g_hash_table_insert(groups, group, hits);
    }
    else
      g_free(group);

    const _tag_hit_t hit = { tagid, score };
    g_array_append_val(hits, hit);
  }
  sqlite3_finalize(stmt);
  g_free(embedding);

  int applied = 0;
  GHashTableIter iter;
  gpointer key, value;
  g_hash_table_iter_init(&iter, groups);
  while(g_hash_table_iter_next(&iter, &key, &value))
  {
    GArray *hits = value;
    g_array_sort(hits, _hit_cmp);
    const guint k = MIN((guint)TAG_TOP_K_PER_GROUP, hits->len);
    for(guint i = 0; i < k; i++)
    {
      const int tagid = g_array_index(hits, _tag_hit_t, i).tagid;
      dt_tag_attach(tagid, imgid, FALSE, FALSE);

      // remember it was us, not the user
      sqlite3_stmt *rec = NULL;
      DT_DEBUG_SQLITE3_PREPARE_V2(db,
        "INSERT OR IGNORE INTO embed.auto_tagged (imgid, tagid)"
        "  VALUES (?1, ?2)", -1, &rec, NULL);
      DT_DEBUG_SQLITE3_BIND_INT(rec, 1, imgid);
      DT_DEBUG_SQLITE3_BIND_INT(rec, 2, tagid);
      sqlite3_step(rec);
      sqlite3_finalize(rec);
      applied++;
    }
  }
  g_hash_table_destroy(groups);

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

  dt_database_start_transaction(darktable.db);
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
  dt_database_release_transaction(darktable.db);

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
