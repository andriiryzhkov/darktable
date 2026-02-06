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

#include "common/ai_models.h"
#include "common/darktable.h"
#include "common/curl_tools.h"
#include "common/file_location.h"
#include "control/control.h"
#include "control/jobs.h"

#include <archive.h>
#include <archive_entry.h>
#include <curl/curl.h>
#include <json-glib/json-glib.h>
#include <string.h>

// Config keys
#define CONF_AI_ENABLED "plugins/ai/enabled"
#define CONF_AI_PROVIDER "plugins/ai/provider"
#define CONF_AI_REPOSITORY "plugins/ai/repository"
#define CONF_AI_RELEASE "plugins/ai/release"
#define CONF_MODEL_ENABLED_PREFIX "plugins/ai/models/"

// --- Internal Helpers ---

static void _model_free(dt_ai_model_t *model)
{
  if(!model) return;
  g_free(model->id);
  g_free(model->name);
  g_free(model->description);
  g_free(model->task);
  g_free(model->github_asset);
  g_free(model->checksum);
  g_free(model);
}

static dt_ai_model_t *_model_new(void)
{
  dt_ai_model_t *model = g_new0(dt_ai_model_t, 1);
  model->enabled = TRUE;
  model->status = DT_AI_MODEL_NOT_DOWNLOADED;
  return model;
}

static gboolean _ensure_directory(const char *path)
{
  if(g_file_test(path, G_FILE_TEST_IS_DIR))
    return TRUE;
  return g_mkdir_with_parents(path, 0755) == 0;
}

// --- Provider Helpers ---

const char *dt_ai_provider_to_string(dt_ai_provider_t provider)
{
  switch(provider)
  {
    case DT_AI_PROVIDER_CPU:      return "CPU";
    case DT_AI_PROVIDER_COREML:   return "CoreML";
    case DT_AI_PROVIDER_CUDA:     return "CUDA";
    case DT_AI_PROVIDER_ROCM:     return "ROCm";
    case DT_AI_PROVIDER_DIRECTML: return "DirectML";
    default:                      return "auto";
  }
}

dt_ai_provider_t dt_ai_provider_from_string(const char *str)
{
  if(!str) return DT_AI_PROVIDER_AUTO;
  if(g_ascii_strcasecmp(str, "CPU") == 0) return DT_AI_PROVIDER_CPU;
  if(g_ascii_strcasecmp(str, "CoreML") == 0) return DT_AI_PROVIDER_COREML;
  if(g_ascii_strcasecmp(str, "CUDA") == 0) return DT_AI_PROVIDER_CUDA;
  if(g_ascii_strcasecmp(str, "ROCm") == 0) return DT_AI_PROVIDER_ROCM;
  if(g_ascii_strcasecmp(str, "DirectML") == 0) return DT_AI_PROVIDER_DIRECTML;
  return DT_AI_PROVIDER_AUTO;
}

// --- Core API ---

dt_ai_registry_t *dt_ai_models_init(void)
{
  dt_ai_registry_t *registry = g_new0(dt_ai_registry_t, 1);
  g_mutex_init(&registry->lock);

  // Set up directories
  char cachedir[PATH_MAX] = { 0 };
  dt_loc_get_user_cache_dir(cachedir, sizeof(cachedir));

  // Models go in XDG data dir (~/.local/share/darktable/models)
  registry->models_dir = g_build_filename(g_get_user_data_dir(), "darktable", "models", NULL);
  registry->cache_dir = g_build_filename(cachedir, "ai_downloads", NULL);

  // Ensure directories exist
  _ensure_directory(registry->models_dir);
  _ensure_directory(registry->cache_dir);

  // Load settings from config
  registry->ai_enabled = dt_conf_get_bool(CONF_AI_ENABLED);

  char *provider_str = dt_conf_get_string(CONF_AI_PROVIDER);
  registry->provider = dt_ai_provider_from_string(provider_str);
  g_free(provider_str);

  dt_print(DT_DEBUG_AI, "[ai_models] Initialized: models_dir=%s, cache_dir=%s",
           registry->models_dir, registry->cache_dir);

  return registry;
}

static dt_ai_model_t *_parse_model_json(JsonObject *obj)
{
  if(!json_object_has_member(obj, "id") || !json_object_has_member(obj, "name"))
    return NULL;

  dt_ai_model_t *model = _model_new();
  model->id = g_strdup(json_object_get_string_member(obj, "id"));
  model->name = g_strdup(json_object_get_string_member(obj, "name"));

  if(json_object_has_member(obj, "description"))
    model->description = g_strdup(json_object_get_string_member(obj, "description"));
  if(json_object_has_member(obj, "task"))
    model->task = g_strdup(json_object_get_string_member(obj, "task"));
  if(json_object_has_member(obj, "github_asset"))
    model->github_asset = g_strdup(json_object_get_string_member(obj, "github_asset"));
  if(json_object_has_member(obj, "checksum"))
    model->checksum = g_strdup(json_object_get_string_member(obj, "checksum"));
  if(json_object_has_member(obj, "required"))
    model->required = json_object_get_boolean_member(obj, "required");

  return model;
}

gboolean dt_ai_models_load_registry(dt_ai_registry_t *registry)
{
  if(!registry) return FALSE;

  // Find the registry JSON file in the data directory
  char datadir[PATH_MAX] = { 0 };
  dt_loc_get_datadir(datadir, sizeof(datadir));
  char *registry_path = g_build_filename(datadir, "ai_models.json", NULL);

  if(!g_file_test(registry_path, G_FILE_TEST_EXISTS))
  {
    dt_print(DT_DEBUG_AI, "[ai_models] Registry file not found: %s", registry_path);
    g_free(registry_path);
    return FALSE;
  }

  GError *error = NULL;
  JsonParser *parser = json_parser_new();

  if(!json_parser_load_from_file(parser, registry_path, &error))
  {
    dt_print(DT_DEBUG_AI, "[ai_models] Failed to parse registry: %s",
             error ? error->message : "unknown error");
    if(error) g_error_free(error);
    g_object_unref(parser);
    g_free(registry_path);
    return FALSE;
  }

  JsonNode *root = json_parser_get_root(parser);
  if(!JSON_NODE_HOLDS_OBJECT(root))
  {
    dt_print(DT_DEBUG_AI, "[ai_models] Registry root is not an object");
    g_object_unref(parser);
    g_free(registry_path);
    return FALSE;
  }

  JsonObject *root_obj = json_node_get_object(root);

  g_mutex_lock(&registry->lock);

  // Clear existing models
  g_list_free_full(registry->models, (GDestroyNotify)_model_free);
  registry->models = NULL;

  // Parse repository and release - config overrides JSON defaults
  g_free(registry->repository);
  g_free(registry->release);
  registry->repository = NULL;
  registry->release = NULL;

  // Get defaults from JSON
  const char *json_repository = NULL;
  const char *json_release = NULL;
  if(json_object_has_member(root_obj, "repository"))
    json_repository = json_object_get_string_member(root_obj, "repository");
  if(json_object_has_member(root_obj, "release"))
    json_release = json_object_get_string_member(root_obj, "release");

  // Use config values if set, otherwise use JSON defaults
  if(dt_conf_key_exists(CONF_AI_REPOSITORY))
    registry->repository = dt_conf_get_string(CONF_AI_REPOSITORY);
  else if(json_repository)
    registry->repository = g_strdup(json_repository);

  if(dt_conf_key_exists(CONF_AI_RELEASE))
    registry->release = dt_conf_get_string(CONF_AI_RELEASE);
  else if(json_release)
    registry->release = g_strdup(json_release);

  dt_print(DT_DEBUG_AI, "[ai_models] Using repository: %s, release: %s",
           registry->repository ? registry->repository : "(none)",
           registry->release ? registry->release : "(none)");

  // Parse models array
  if(json_object_has_member(root_obj, "models"))
  {
    JsonArray *models_arr = json_object_get_array_member(root_obj, "models");
    guint len = json_array_get_length(models_arr);

    for(guint i = 0; i < len; i++)
    {
      JsonNode *node = json_array_get_element(models_arr, i);
      if(!JSON_NODE_HOLDS_OBJECT(node)) continue;

      dt_ai_model_t *model = _parse_model_json(json_node_get_object(node));
      if(model)
      {
        // Load enabled state from user config
        char *conf_key = g_strdup_printf("%s%s/enabled", CONF_MODEL_ENABLED_PREFIX, model->id);
        if(dt_conf_key_exists(conf_key))
          model->enabled = dt_conf_get_bool(conf_key);
        g_free(conf_key);

        registry->models = g_list_prepend(registry->models, model);
        dt_print(DT_DEBUG_AI, "[ai_models] Loaded model: %s (%s)", model->name, model->id);
      }
    }
  }

  // Reverse to restore original JSON order (we used prepend for O(1) insertion)
  registry->models = g_list_reverse(registry->models);

  g_mutex_unlock(&registry->lock);

  dt_print(DT_DEBUG_AI, "[ai_models] Registry loaded: %d models from %s",
           g_list_length(registry->models), registry_path);

  g_object_unref(parser);
  g_free(registry_path);

  // Check which models are actually downloaded
  dt_ai_models_refresh_status(registry);

  return TRUE;
}

void dt_ai_models_refresh_status(dt_ai_registry_t *registry)
{
  if(!registry) return;

  g_mutex_lock(&registry->lock);

  for(GList *l = registry->models; l; l = g_list_next(l))
  {
    dt_ai_model_t *model = (dt_ai_model_t *)l->data;

    // Check if model directory exists and contains required files
    char *model_dir = g_build_filename(registry->models_dir, model->id, NULL);
    char *config_path = g_build_filename(model_dir, "config.json", NULL);

    if(g_file_test(model_dir, G_FILE_TEST_IS_DIR) &&
       g_file_test(config_path, G_FILE_TEST_EXISTS))
    {
      model->status = DT_AI_MODEL_DOWNLOADED;
    }
    else
    {
      model->status = DT_AI_MODEL_NOT_DOWNLOADED;
    }

    g_free(config_path);
    g_free(model_dir);
  }

  g_mutex_unlock(&registry->lock);
}

void dt_ai_models_cleanup(dt_ai_registry_t *registry)
{
  if(!registry) return;

  g_mutex_lock(&registry->lock);
  g_list_free_full(registry->models, (GDestroyNotify)_model_free);
  registry->models = NULL;
  g_mutex_unlock(&registry->lock);

  g_mutex_clear(&registry->lock);

  g_free(registry->repository);
  g_free(registry->release);
  g_free(registry->models_dir);
  g_free(registry->cache_dir);
  g_free(registry);
}

// --- Model Access ---

int dt_ai_models_get_count(dt_ai_registry_t *registry)
{
  if(!registry) return 0;
  g_mutex_lock(&registry->lock);
  int count = g_list_length(registry->models);
  g_mutex_unlock(&registry->lock);
  return count;
}

dt_ai_model_t *dt_ai_models_get_by_index(dt_ai_registry_t *registry, int index)
{
  if(!registry || index < 0) return NULL;
  g_mutex_lock(&registry->lock);
  dt_ai_model_t *model = g_list_nth_data(registry->models, index);
  g_mutex_unlock(&registry->lock);
  return model;
}

dt_ai_model_t *dt_ai_models_get_by_id(dt_ai_registry_t *registry, const char *model_id)
{
  if(!registry || !model_id) return NULL;

  g_mutex_lock(&registry->lock);
  dt_ai_model_t *result = NULL;
  for(GList *l = registry->models; l; l = g_list_next(l))
  {
    dt_ai_model_t *model = (dt_ai_model_t *)l->data;
    if(g_strcmp0(model->id, model_id) == 0)
    {
      result = model;
      break;
    }
  }
  g_mutex_unlock(&registry->lock);
  return result;
}

// --- Download Implementation ---

typedef struct dt_ai_download_data_t {
  dt_ai_registry_t *registry;
  dt_ai_model_t *model;
  dt_ai_progress_callback callback;
  gpointer user_data;
  FILE *file;
  const gboolean *cancel_flag;  // Optional: set to non-NULL to enable cancellation
} dt_ai_download_data_t;

static size_t _curl_write_callback(void *ptr, size_t size, size_t nmemb, void *data)
{
  dt_ai_download_data_t *dl = (dt_ai_download_data_t *)data;
  return fwrite(ptr, size, nmemb, dl->file);
}

static int _curl_progress_callback(void *clientp, curl_off_t dltotal, curl_off_t dlnow,
                                   curl_off_t ultotal, curl_off_t ulnow)
{
  dt_ai_download_data_t *dl = (dt_ai_download_data_t *)clientp;

  // Check for cancellation
  if(dl->cancel_flag && g_atomic_int_get(dl->cancel_flag))
    return 1;  // Non-zero aborts the transfer

  if(dltotal > 0)
  {
    double progress = (double)dlnow / (double)dltotal;
    dl->model->download_progress = progress;
    if(dl->callback)
      dl->callback(dl->model->id, progress, dl->user_data);
  }
  return 0;
}

static gboolean _verify_checksum(const char *filepath, const char *expected)
{
  if(!expected || !g_str_has_prefix(expected, "sha256:"))
  {
    dt_print(DT_DEBUG_AI, "[ai_models] No valid checksum provided - rejecting download");
    return FALSE;  // Reject files without a valid checksum
  }

  const char *expected_hash = expected + 7;  // Skip "sha256:"

  GChecksum *checksum = g_checksum_new(G_CHECKSUM_SHA256);
  if(!checksum) return FALSE;

  // Stream file in chunks to avoid loading entire file into memory
  FILE *f = g_fopen(filepath, "rb");
  if(!f)
  {
    dt_print(DT_DEBUG_AI, "[ai_models] Failed to open file for checksum: %s", filepath);
    g_checksum_free(checksum);
    return FALSE;
  }

  guchar buf[65536];
  size_t n;
  while((n = fread(buf, 1, sizeof(buf), f)) > 0)
    g_checksum_update(checksum, buf, n);
  fclose(f);

  const gchar *computed = g_checksum_get_string(checksum);
  gboolean match = g_ascii_strcasecmp(computed, expected_hash) == 0;

  if(!match)
  {
    dt_print(DT_DEBUG_AI, "[ai_models] Checksum mismatch: expected %s, got %s",
             expected_hash, computed);
  }

  g_checksum_free(checksum);
  return match;
}

static gboolean _extract_zip(const char *zippath, const char *destdir)
{
  struct archive *a = archive_read_new();
  struct archive *ext = archive_write_disk_new();
  struct archive_entry *entry;
  int r;
  gboolean success = TRUE;

  archive_read_support_format_zip(a);
  archive_write_disk_set_options(ext, ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM
                                     | ARCHIVE_EXTRACT_SECURE_SYMLINKS
                                     | ARCHIVE_EXTRACT_SECURE_NODOTDOT);

  if((r = archive_read_open_filename(a, zippath, 10240)) != ARCHIVE_OK)
  {
    dt_print(DT_DEBUG_AI, "[ai_models] Failed to open archive: %s", archive_error_string(a));
    archive_read_free(a);
    archive_write_free(ext);
    return FALSE;
  }

  _ensure_directory(destdir);

  // Resolve destdir to a canonical path for path traversal validation
  char *real_destdir = realpath(destdir, NULL);
  if(!real_destdir)
  {
    dt_print(DT_DEBUG_AI, "[ai_models] Failed to resolve destdir: %s", destdir);
    archive_read_close(a);
    archive_read_free(a);
    archive_write_free(ext);
    return FALSE;
  }
  const size_t destdir_len = strlen(real_destdir);

  while(archive_read_next_header(a, &entry) == ARCHIVE_OK)
  {
    const char *entry_name = archive_entry_pathname(entry);

    // Reject entries with path traversal components
    if(g_strstr_len(entry_name, -1, "..") != NULL)
    {
      dt_print(DT_DEBUG_AI, "[ai_models] Skipping suspicious archive entry: %s", entry_name);
      continue;
    }

    // Build full path in destination
    char *full_path = g_build_filename(real_destdir, entry_name, NULL);

    // Verify the resolved path is within destdir
    char *real_full_path = realpath(full_path, NULL);
    // For new files, realpath returns NULL; check the parent directory instead
    if(!real_full_path)
    {
      char *parent = g_path_get_dirname(full_path);
      _ensure_directory(parent);
      char *real_parent = realpath(parent, NULL);
      g_free(parent);
      if(!real_parent || strncmp(real_parent, real_destdir, destdir_len) != 0)
      {
        dt_print(DT_DEBUG_AI, "[ai_models] Path traversal blocked: %s", entry_name);
        free(real_parent);
        g_free(full_path);
        continue;
      }
      free(real_parent);
    }
    else
    {
      if(strncmp(real_full_path, real_destdir, destdir_len) != 0)
      {
        dt_print(DT_DEBUG_AI, "[ai_models] Path traversal blocked: %s", entry_name);
        free(real_full_path);
        g_free(full_path);
        continue;
      }
      free(real_full_path);
    }

    archive_entry_set_pathname(entry, full_path);

    r = archive_write_header(ext, entry);
    if(r == ARCHIVE_OK)
    {
      const void *buff;
      size_t size;
      la_int64_t offset;

      while(archive_read_data_block(a, &buff, &size, &offset) == ARCHIVE_OK)
      {
        if(archive_write_data_block(ext, buff, size, offset) != ARCHIVE_OK)
        {
          dt_print(DT_DEBUG_AI, "[ai_models] Write error: %s", archive_error_string(ext));
          success = FALSE;
          break;
        }
      }
      if(archive_write_finish_entry(ext) != ARCHIVE_OK)
        success = FALSE;
    }
    else
    {
      dt_print(DT_DEBUG_AI, "[ai_models] Write header error: %s", archive_error_string(ext));
      success = FALSE;
    }

    g_free(full_path);
  }

  free(real_destdir);
  archive_read_close(a);
  archive_read_free(a);
  archive_write_close(ext);
  archive_write_free(ext);

  return success;
}

// Synchronous download - returns error message or NULL on success
char *dt_ai_models_download_sync(dt_ai_registry_t *registry, const char *model_id,
                                  dt_ai_progress_callback callback, gpointer user_data,
                                  const gboolean *cancel_flag)
{
  dt_print(DT_DEBUG_AI, "[ai_models] Download requested for: %s", model_id ? model_id : "(null)");

  if(!registry || !model_id)
    return g_strdup(_("invalid parameters"));

  dt_ai_model_t *model = dt_ai_models_get_by_id(registry, model_id);
  if(!model)
    return g_strdup(_("model not found in registry"));

  if(!model->github_asset)
    return g_strdup(_("model has no download asset defined"));

  // Validate asset filename: reject path separators and query strings
  if(strchr(model->github_asset, '/') || strchr(model->github_asset, '\\')
     || strchr(model->github_asset, '?') || strchr(model->github_asset, '#')
     || strstr(model->github_asset, ".."))
    return g_strdup(_("invalid asset filename"));

  g_mutex_lock(&registry->lock);
  if(model->status == DT_AI_MODEL_DOWNLOADING)
  {
    g_mutex_unlock(&registry->lock);
    return g_strdup(_("model is already downloading"));
  }
  model->status = DT_AI_MODEL_DOWNLOADING;
  model->download_progress = 0.0;
  g_mutex_unlock(&registry->lock);

  // Validate repository format (must be "owner/repo" with safe characters)
  if(!registry->repository || !registry->release
     || !g_regex_match_simple("^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$", registry->repository, 0, 0)
     || !g_regex_match_simple("^[a-zA-Z0-9._-]+$", registry->release, 0, 0))
  {
    g_mutex_lock(&registry->lock);
    model->status = DT_AI_MODEL_ERROR;
    g_mutex_unlock(&registry->lock);
    return g_strdup(_("invalid repository or release format"));
  }

  // Build GitHub download URL
  char *url = NULL;
  if(g_strcmp0(registry->release, "latest") == 0)
  {
    url = g_strdup_printf("https://github.com/%s/releases/latest/download/%s",
                          registry->repository, model->github_asset);
  }
  else
  {
    url = g_strdup_printf("https://github.com/%s/releases/download/%s/%s",
                          registry->repository, registry->release,
                          model->github_asset);
  }

  if(!url)
  {
    g_mutex_lock(&registry->lock);
    model->status = DT_AI_MODEL_ERROR;
    g_mutex_unlock(&registry->lock);
    return g_strdup(_("failed to build download URL"));
  }

  dt_print(DT_DEBUG_AI, "[ai_models] Downloading: %s", url);

  // Prepare download path
  char *download_path = g_build_filename(registry->cache_dir, model->github_asset, NULL);

  FILE *file = g_fopen(download_path, "wb");
  if(!file)
  {
    char *err = g_strdup_printf(_("failed to create file: %s"), download_path);
    g_free(download_path);
    g_free(url);
    g_mutex_lock(&registry->lock);
    model->status = DT_AI_MODEL_ERROR;
    g_mutex_unlock(&registry->lock);
    return err;
  }

  // Set up download data
  dt_ai_download_data_t dl = {
    .registry = registry,
    .model = model,
    .callback = callback,
    .user_data = user_data,
    .file = file,
    .cancel_flag = cancel_flag
  };

  // Initialize curl
  CURL *curl = curl_easy_init();
  if(!curl)
  {
    fclose(file);
    g_free(download_path);
    g_free(url);
    g_mutex_lock(&registry->lock);
    model->status = DT_AI_MODEL_ERROR;
    g_mutex_unlock(&registry->lock);
    return g_strdup(_("failed to initialize download"));
  }
  dt_curl_init(curl, FALSE);

  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _curl_write_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &dl);
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, _curl_progress_callback);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &dl);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

  CURLcode res = curl_easy_perform(curl);

  fclose(file);

  char *error = NULL;

  if(res != CURLE_OK)
  {
    if(res == CURLE_ABORTED_BY_CALLBACK)
      error = g_strdup(_("download cancelled"));
    else
      error = g_strdup_printf(_("download failed: %s"), curl_easy_strerror(res));
    g_unlink(download_path);
  }
  else
  {
    // Check HTTP response code
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if(http_code != 200)
    {
      error = g_strdup_printf(_("HTTP error: %ld"), http_code);
      g_unlink(download_path);
    }
  }

  curl_easy_cleanup(curl);
  g_free(url);

  if(error)
  {
    g_free(download_path);
    g_mutex_lock(&registry->lock);
    model->status = DT_AI_MODEL_ERROR;
    g_mutex_unlock(&registry->lock);
    return error;
  }

  // Verify checksum
  if(!_verify_checksum(download_path, model->checksum))
  {
    g_unlink(download_path);
    g_free(download_path);
    g_mutex_lock(&registry->lock);
    model->status = DT_AI_MODEL_ERROR;
    g_mutex_unlock(&registry->lock);
    return g_strdup(_("checksum verification failed"));
  }

  // Extract to models directory (ZIP already contains model_id folder)
  if(!_extract_zip(download_path, registry->models_dir))
  {
    g_unlink(download_path);
    g_free(download_path);
    g_mutex_lock(&registry->lock);
    model->status = DT_AI_MODEL_ERROR;
    g_mutex_unlock(&registry->lock);
    return g_strdup(_("failed to extract archive"));
  }

  // Clean up downloaded zip
  g_unlink(download_path);
  g_free(download_path);

  g_mutex_lock(&registry->lock);
  model->status = DT_AI_MODEL_DOWNLOADED;
  model->download_progress = 1.0;
  g_mutex_unlock(&registry->lock);

  dt_print(DT_DEBUG_AI, "[ai_models] Download complete: %s", model->id);

  // Final callback
  if(callback)
    callback(model_id, 1.0, user_data);

  return NULL;  // Success
}

// Wrapper that returns boolean for compatibility
gboolean dt_ai_models_download(dt_ai_registry_t *registry, const char *model_id,
                               dt_ai_progress_callback callback, gpointer user_data)
{
  char *error = dt_ai_models_download_sync(registry, model_id, callback, user_data, NULL);
  if(error)
  {
    dt_print(DT_DEBUG_AI, "[ai_models] Download error: %s", error);
    g_free(error);
    return FALSE;
  }
  return TRUE;
}

gboolean dt_ai_models_download_required(dt_ai_registry_t *registry,
                                        dt_ai_progress_callback callback,
                                        gpointer user_data)
{
  if(!registry) return FALSE;

  // Collect IDs while holding lock, then download without lock
  GList *ids = NULL;
  g_mutex_lock(&registry->lock);
  for(GList *l = registry->models; l; l = g_list_next(l))
  {
    dt_ai_model_t *model = (dt_ai_model_t *)l->data;
    if(model->required && model->status == DT_AI_MODEL_NOT_DOWNLOADED)
      ids = g_list_prepend(ids, g_strdup(model->id));
  }
  g_mutex_unlock(&registry->lock);

  gboolean any_started = FALSE;
  for(GList *l = ids; l; l = g_list_next(l))
  {
    if(dt_ai_models_download(registry, (const char *)l->data, callback, user_data))
      any_started = TRUE;
  }
  g_list_free_full(ids, g_free);
  return any_started;
}

gboolean dt_ai_models_download_all(dt_ai_registry_t *registry,
                                   dt_ai_progress_callback callback,
                                   gpointer user_data)
{
  if(!registry) return FALSE;

  // Collect IDs while holding lock, then download without lock
  GList *ids = NULL;
  g_mutex_lock(&registry->lock);
  for(GList *l = registry->models; l; l = g_list_next(l))
  {
    dt_ai_model_t *model = (dt_ai_model_t *)l->data;
    if(model->status == DT_AI_MODEL_NOT_DOWNLOADED)
      ids = g_list_prepend(ids, g_strdup(model->id));
  }
  g_mutex_unlock(&registry->lock);

  gboolean any_started = FALSE;
  for(GList *l = ids; l; l = g_list_next(l))
  {
    if(dt_ai_models_download(registry, (const char *)l->data, callback, user_data))
      any_started = TRUE;
  }
  g_list_free_full(ids, g_free);
  return any_started;
}

static gboolean _rmdir_recursive(const char *path)
{
  if(!g_file_test(path, G_FILE_TEST_IS_DIR))
  {
    g_unlink(path);
    return TRUE;
  }

  GDir *dir = g_dir_open(path, 0, NULL);
  if(!dir) return FALSE;

  const gchar *name;
  while((name = g_dir_read_name(dir)))
  {
    char *child = g_build_filename(path, name, NULL);
    if(g_file_test(child, G_FILE_TEST_IS_SYMLINK))
      g_unlink(child);  // Remove the symlink itself, never follow
    else if(g_file_test(child, G_FILE_TEST_IS_DIR))
      _rmdir_recursive(child);
    else
      g_unlink(child);
    g_free(child);
  }
  g_dir_close(dir);
  return g_rmdir(path) == 0;
}

gboolean dt_ai_models_delete(dt_ai_registry_t *registry, const char *model_id)
{
  if(!registry || !model_id) return FALSE;

  dt_ai_model_t *model = dt_ai_models_get_by_id(registry, model_id);
  if(!model) return FALSE;

  char *model_dir = g_build_filename(registry->models_dir, model_id, NULL);
  _rmdir_recursive(model_dir);
  g_free(model_dir);

  g_mutex_lock(&registry->lock);
  model->status = DT_AI_MODEL_NOT_DOWNLOADED;
  model->download_progress = 0.0;
  g_mutex_unlock(&registry->lock);

  return TRUE;
}

// --- Configuration ---

void dt_ai_models_set_enabled(dt_ai_registry_t *registry, const char *model_id,
                              gboolean enabled)
{
  if(!registry || !model_id) return;

  dt_ai_model_t *model = dt_ai_models_get_by_id(registry, model_id);
  if(!model) return;

  g_mutex_lock(&registry->lock);
  model->enabled = enabled;
  g_mutex_unlock(&registry->lock);

  // Persist to config
  char *conf_key = g_strdup_printf("%s%s/enabled", CONF_MODEL_ENABLED_PREFIX, model_id);
  dt_conf_set_bool(conf_key, enabled);
  g_free(conf_key);
}

char *dt_ai_models_get_path(dt_ai_registry_t *registry, const char *model_id)
{
  if(!registry || !model_id) return NULL;

  dt_ai_model_t *model = dt_ai_models_get_by_id(registry, model_id);
  if(!model || model->status != DT_AI_MODEL_DOWNLOADED)
    return NULL;

  return g_build_filename(registry->models_dir, model_id, NULL);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
