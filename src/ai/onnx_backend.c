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

#include "onnx_backend.h"
#include "common/darktable.h"
#include <glib.h>
#include <json-glib/json-glib.h>
#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// --- Internal Structures ---

struct dt_ai_environment_t {
  GList *models;           // List of dt_ai_model_info_t*
  GHashTable *model_paths; // ID -> Path (string)

  // To keep pointers in dt_ai_model_info_t valid
  GList *string_storage; // List of char*

  // Remembered for refresh
  char *search_paths;
};

struct dt_ai_context_t {
  char *loaded_model_id;

  // ONNX Runtime C Objects
  const OrtApi *ort_api;
  OrtSession *session;
  OrtEnv *env;
  OrtMemoryInfo *memory_info;

  // IO Names
  OrtAllocator *allocator;
  char **input_names;
  char **output_names;
  size_t input_count;
  dt_ai_dtype_t *input_types;
  size_t output_count;
  dt_ai_dtype_t *output_types;
};

// Global pointer to the API struct (loaded once)
static const OrtApi *g_ort = NULL;

// --- Helper Functions ---

// Float16 Conversion Utilities
// Based on: https://gist.github.com/rygorous/2156668

// Robust Float16 Conversion Utilities
// Handles Zero, Denormals, and Infinity correctly.

static uint16_t _float_to_half(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 31) & 1;
  uint32_t exp = (x >> 23) & 0xFF;
  uint32_t mant = x & 0x7FFFFF;

  // Handle Zero and float32 denormals (too small for float16)
  if (exp == 0)
    return (uint16_t)(sign << 15);

  // Handle Infinity / NaN
  if (exp == 255)
    return (uint16_t)((sign << 15) | 0x7C00 | (mant ? 1 : 0));

  // Re-bias exponent from float32 (bias 127) to float16 (bias 15)
  int new_exp = (int)exp - 127 + 15;

  if (new_exp <= 0) {
    // Encode as float16 denormal: shift mantissa with implicit leading 1
    // The implicit 1 bit plus 10 mantissa bits, shifted right by (1 - new_exp)
    int shift = 1 - new_exp;
    if (shift > 24) return (uint16_t)(sign << 15); // too small even for denormal
    uint32_t full_mant = (1 << 23) | mant; // restore implicit leading 1
    uint16_t half_mant = (uint16_t)(full_mant >> (13 + shift));
    return (uint16_t)((sign << 15) | half_mant);
  } else if (new_exp >= 31) {
    // Overflow to Infinity
    return (uint16_t)((sign << 15) | 0x7C00);
  }

  return (uint16_t)((sign << 15) | (new_exp << 10) | (mant >> 13));
}

static float _half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0) {
      // Zero
      uint32_t result = (sign << 31);
      float f;
      memcpy(&f, &result, 4);
      return f;
    }
    // Denormal: value = (-1)^sign * 2^(-14) * (mant / 1024)
    // Convert to float32 by normalizing: find leading 1 and shift
    uint32_t m = mant;
    int e = -1;
    while (!(m & 0x400)) { // shift until leading 1 reaches bit 10
      m <<= 1;
      e--;
    }
    m &= 0x3FF; // remove the leading 1
    uint32_t new_exp = (uint32_t)(e + 127 - 14 + 1);
    uint32_t result = (sign << 31) | (new_exp << 23) | (m << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
  } else if (exp == 31) {
    // Inf / NaN
    uint32_t result = (sign << 31) | 0x7F800000 | (mant << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
  }

  // Normalized
  uint32_t new_exp = exp + 127 - 15;
  uint32_t result = (sign << 31) | (new_exp << 23) | (mant << 13);
  float f;
  memcpy(&f, &result, sizeof(f));
  return f;
}

static void _store_string(dt_ai_environment_t *env, const char *str,
                          const char **out_ptr) {
  char *copy = g_strdup(str);
  env->string_storage = g_list_prepend(env->string_storage, copy);
  *out_ptr = copy;
}

static void _scan_directory(dt_ai_environment_t *env, const char *root_path) {
  GDir *dir = g_dir_open(root_path, 0, NULL);
  if (!dir)
    return;

  const char *entry_name;
  while ((entry_name = g_dir_read_name(dir))) {
    char *full_path = g_build_filename(root_path, entry_name, NULL);
    if (g_file_test(full_path, G_FILE_TEST_IS_DIR)) {
      char *config_path = g_build_filename(full_path, "config.json", NULL);

      if (g_file_test(config_path, G_FILE_TEST_EXISTS)) {
        JsonParser *parser = json_parser_new();
        GError *error = NULL;

        if (json_parser_load_from_file(parser, config_path, &error)) {
          JsonNode *root = json_parser_get_root(parser);
          JsonObject *obj = json_node_get_object(root);

          const char *id = json_object_get_string_member(obj, "id");
          const char *name = json_object_get_string_member(obj, "name");
          const char *desc =
              json_object_has_member(obj, "description")
                  ? json_object_get_string_member(obj, "description")
                  : "";
          const char *task = json_object_has_member(obj, "task")
                                 ? json_object_get_string_member(obj, "task")
                                 : "general";

          if (id && name) {
            // Skip duplicate model IDs (first discovered wins)
            if (g_hash_table_contains(env->model_paths, id)) {
              dt_print(DT_DEBUG_AI,
                       "[darktable_ai] Skipping duplicate model ID: %s", id);
            } else {
              dt_ai_model_info_t *info = g_new0(dt_ai_model_info_t, 1);
              _store_string(env, id, &info->id);
              _store_string(env, name, &info->name);
              _store_string(env, desc, &info->description);
              _store_string(env, task, &info->task_type);
              info->num_inputs = json_object_has_member(obj, "num_inputs")
                  ? (int)json_object_get_int_member(obj, "num_inputs")
                  : 1;

              env->models = g_list_append(env->models, info);
              g_hash_table_insert(env->model_paths, g_strdup(info->id),
                                  g_strdup(full_path));

              dt_print(DT_DEBUG_AI, "[darktable_ai] Discovered: %s (%s)", name,
                       id);
            }
          }
        } else {
          dt_print(DT_DEBUG_AI, "[darktable_ai] Parse error: %s",
                   error->message);
          g_error_free(error);
        }
        g_object_unref(parser);
      }
      g_free(config_path);
    }
    g_free(full_path);
  }
  g_dir_close(dir);
}

// --- API Implementation ---

DT_AI_EXPORT dt_ai_environment_t *dt_ai_env_init(const char *search_paths) {
  dt_print(DT_DEBUG_AI, "[darktable_ai] dt_ai_env_init start.");

  if (!g_ort) {
    dt_print(DT_DEBUG_AI, "[darktable_ai] Initializing ORT API");
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
      dt_print(DT_DEBUG_AI, "[darktable_ai] Failed to init ONNX Runtime API");
      return NULL;
    }
  }

  dt_ai_environment_t *env = g_new0(dt_ai_environment_t, 1);
  env->model_paths =
      g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
  env->search_paths = g_strdup(search_paths);

  // Parse Paths
  if (search_paths) {
    char **tokens = g_strsplit(search_paths, ";", -1);
    for (int i = 0; tokens[i] != NULL; i++) {
      _scan_directory(env, tokens[i]);
    }
    g_strfreev(tokens);
  }

  // Default Paths
  // Priority 1: ~/.config/darktable/models (or User Config Dir)
  const char *config_dir = g_get_user_config_dir();
  if (config_dir) {
    char *p = g_build_filename(config_dir, "darktable", "models", NULL);
    _scan_directory(env, p);
    g_free(p);
  }

  // Priority 2: ~/.local/share/darktable/models (or User Data Dir)
  const char *data_dir = g_get_user_data_dir();
  if (data_dir) {
    char *p = g_build_filename(data_dir, "darktable", "models", NULL);
    _scan_directory(env, p);
    g_free(p);
  }

  return env;
}

DT_AI_EXPORT int dt_ai_get_model_count(dt_ai_environment_t *env) {
  if (!env)
    return 0;
  return g_list_length(env->models);
}

DT_AI_EXPORT const dt_ai_model_info_t *
dt_ai_get_model_info_by_index(dt_ai_environment_t *env, int index) {
  if (!env)
    return NULL;
  GList *item = g_list_nth(env->models, index);
  if (!item)
    return NULL;
  return (const dt_ai_model_info_t *)item->data;
}

DT_AI_EXPORT const dt_ai_model_info_t *
dt_ai_get_model_info_by_id(dt_ai_environment_t *env, const char *id) {
  if (!env || !id)
    return NULL;
  for (GList *l = env->models; l != NULL; l = l->next) {
    dt_ai_model_info_t *info = (dt_ai_model_info_t *)l->data;
    if (strcmp(info->id, id) == 0)
      return info;
  }
  return NULL;
}

static void _free_model_info(gpointer data) { g_free(data); }

DT_AI_EXPORT void dt_ai_env_refresh(dt_ai_environment_t *env) {
  if (!env)
    return;

  dt_print(DT_DEBUG_AI, "[darktable_ai] Refreshing model list");

  // Clear existing data
  g_list_free_full(env->models, _free_model_info);
  env->models = NULL;

  g_list_free_full(env->string_storage, g_free);
  env->string_storage = NULL;

  g_hash_table_remove_all(env->model_paths);

  // Rescan custom paths (if any were provided at init)
  if (env->search_paths) {
    char **tokens = g_strsplit(env->search_paths, ";", -1);
    for (int i = 0; tokens[i] != NULL; i++) {
      _scan_directory(env, tokens[i]);
    }
    g_strfreev(tokens);
  }

  // Rescan directories (same logic as dt_ai_env_init)
  // Priority 1: ~/.config/darktable/models (or User Config Dir)
  const char *config_dir = g_get_user_config_dir();
  if (config_dir) {
    char *p = g_build_filename(config_dir, "darktable", "models", NULL);
    _scan_directory(env, p);
    g_free(p);
  }

  // Priority 2: ~/.local/share/darktable/models (or User Data Dir)
  const char *data_dir = g_get_user_data_dir();
  if (data_dir) {
    char *p = g_build_filename(data_dir, "darktable", "models", NULL);
    _scan_directory(env, p);
    g_free(p);
  }

  dt_print(DT_DEBUG_AI, "[darktable_ai] Refresh complete, found %d models",
           g_list_length(env->models));
}

DT_AI_EXPORT void dt_ai_env_destroy(dt_ai_environment_t *env) {
  if (!env)
    return;

  g_list_free_full(env->models, _free_model_info);
  g_list_free_full(env->string_storage, g_free);
  g_hash_table_destroy(env->model_paths);
  g_free(env->search_paths);

  g_free(env);
}

// --- Optimization Helpers ---

// Try to find and call an ORT execution provider function at runtime via
// dynamic symbol lookup (GModule/dlsym).  Returns TRUE if the provider was
// enabled successfully, FALSE otherwise.
static gboolean _try_provider(OrtSessionOptions *session_opts,
                               const char *symbol_name,
                               const char *provider_name) {
  OrtStatus *status = NULL;
  gboolean ok = FALSE;

  dt_print(DT_DEBUG_AI, "[darktable_ai] Attempting to enable %s...",
           provider_name);

#ifdef _WIN32
  // On Windows, we need to get the handle to onnxruntime.dll, not the main executable
  HMODULE h = GetModuleHandleA("onnxruntime.dll");
  if (!h) {
    // If not already loaded, try to load it
    h = LoadLibraryA("onnxruntime.dll");
  }
  void *func_ptr = NULL;
  if (h) {
    func_ptr = (void *)GetProcAddress(h, symbol_name);
    // Don't call FreeLibrary - we want to keep onnxruntime.dll loaded
  }
#else
  GModule *mod = g_module_open(NULL, 0);
  void *func_ptr = NULL;
  g_module_symbol(mod, symbol_name, &func_ptr);
#endif

  if (func_ptr) {
    // All provider append functions take (OrtSessionOptions*, uint32_t/int)
    typedef OrtStatus *(*ProviderAppender)(OrtSessionOptions *, uint32_t);
    ProviderAppender appender = (ProviderAppender)func_ptr;
    status = appender(session_opts, 0);
    if (!status) {
      dt_print(DT_DEBUG_AI, "[darktable_ai] %s enabled successfully.",
               provider_name);
      ok = TRUE;
    } else {
      dt_print(DT_DEBUG_AI, "[darktable_ai] %s enable failed: %s",
               provider_name, g_ort->GetErrorMessage(status));
      g_ort->ReleaseStatus(status);
    }
  } else {
    dt_print(DT_DEBUG_AI, "[darktable_ai] %s provider not found.",
             provider_name);
  }

#ifndef _WIN32
  g_module_close(mod);
#endif

  return ok;
}

static void _enable_acceleration(OrtSessionOptions *session_opts,
                                 dt_ai_provider_t provider) {
  switch(provider)
  {
    case DT_AI_PROVIDER_CPU:
      // CPU only - don't enable any accelerator
      dt_print(DT_DEBUG_AI, "[darktable_ai] Using CPU only (no hardware acceleration)");
      break;

    case DT_AI_PROVIDER_COREML:
#if defined(__APPLE__)
      _try_provider(session_opts,
                    "OrtSessionOptionsAppendExecutionProvider_CoreML", "CoreML");
#else
      dt_print(DT_DEBUG_AI, "[darktable_ai] CoreML not available on this platform");
#endif
      break;

    case DT_AI_PROVIDER_CUDA:
      _try_provider(session_opts,
                    "OrtSessionOptionsAppendExecutionProvider_CUDA", "CUDA");
      break;

    case DT_AI_PROVIDER_ROCM:
      _try_provider(session_opts,
                    "OrtSessionOptionsAppendExecutionProvider_ROCM", "ROCm");
      break;

    case DT_AI_PROVIDER_DIRECTML:
#if defined(_WIN32)
      _try_provider(session_opts,
                    "OrtSessionOptionsAppendExecutionProvider_DML", "DirectML");
#else
      dt_print(DT_DEBUG_AI, "[darktable_ai] DirectML not available on this platform");
#endif
      break;

    case DT_AI_PROVIDER_AUTO:
    default:
      // Auto-detect best provider based on platform
#if defined(__APPLE__)
      _try_provider(session_opts,
                    "OrtSessionOptionsAppendExecutionProvider_CoreML", "CoreML");
#elif defined(_WIN32)
      _try_provider(session_opts,
                    "OrtSessionOptionsAppendExecutionProvider_DML", "DirectML");
#elif defined(__linux__)
      // Try CUDA first, then ROCm
      if(!_try_provider(session_opts,
                        "OrtSessionOptionsAppendExecutionProvider_CUDA", "CUDA"))
      {
        _try_provider(session_opts,
                      "OrtSessionOptionsAppendExecutionProvider_ROCM", "ROCm");
      }
#endif
      break;
  }
}

// --- Execution ---

DT_AI_EXPORT dt_ai_context_t *dt_ai_load_model(dt_ai_environment_t *env,
                                               const char *model_id,
                                               dt_ai_provider_t provider) {
  if (!env || !model_id)
    return NULL;

  const char *model_dir =
      (const char *)g_hash_table_lookup(env->model_paths, model_id);
  if (!model_dir) {
    dt_print(DT_DEBUG_AI, "[darktable_ai] ID not found: %s", model_id);
    return NULL;
  }

  char *onnx_path = g_build_filename(model_dir, "model.onnx", NULL);
  if (!g_file_test(onnx_path, G_FILE_TEST_EXISTS)) {
    dt_print(DT_DEBUG_AI, "[darktable_ai] Model file missing: %s", onnx_path);
    g_free(onnx_path);
    return NULL;
  }

  dt_print(DT_DEBUG_AI, "[darktable_ai] Loading (C): %s", onnx_path);

  dt_ai_context_t *ctx = g_new0(dt_ai_context_t, 1);
  ctx->loaded_model_id = g_strdup(model_id);
  ctx->ort_api = g_ort;

  // ONNX Init
  OrtStatus *status;

  status =
      g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "DarktableAI", &ctx->env);
  if (status) {
    g_ort->ReleaseStatus(status);
    g_free(onnx_path);
    dt_ai_unload_model(ctx);
    return NULL;
  }

  OrtSessionOptions *session_opts;
  status = g_ort->CreateSessionOptions(&session_opts);
  if (status) {
    g_ort->ReleaseStatus(status);
    dt_ai_unload_model(ctx);
    return NULL;
  }

  // Optimize: Use all available cores (intra-op parallelism)
#ifdef _WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  long num_cores = sysinfo.dwNumberOfProcessors;
#else
  long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  if (num_cores < 1)
    num_cores = 1;

  status = g_ort->SetIntraOpNumThreads(session_opts, (int)num_cores);
  if (status) {
    g_ort->ReleaseStatus(status);
    g_ort->ReleaseSessionOptions(session_opts);
    dt_ai_unload_model(ctx);
    return NULL;
  }

  status =
      g_ort->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL);
  if (status) {
    g_ort->ReleaseStatus(status);
    g_ort->ReleaseSessionOptions(session_opts);
    dt_ai_unload_model(ctx);
    return NULL;
  }

  // Optimize: Enable Hardware Acceleration
  _enable_acceleration(session_opts, provider);

#ifdef _WIN32
  // On Windows, CreateSession expects a wide character string
  wchar_t *onnx_path_wide = (wchar_t *)g_utf8_to_utf16(onnx_path, -1, NULL, NULL, NULL);
  status =
      g_ort->CreateSession(ctx->env, onnx_path_wide, session_opts, &ctx->session);
  g_free(onnx_path_wide);
#else
  status =
      g_ort->CreateSession(ctx->env, onnx_path, session_opts, &ctx->session);
#endif

  g_ort->ReleaseSessionOptions(session_opts);
  g_free(onnx_path);

  if (status) {
    dt_print(DT_DEBUG_AI, "[darktable_ai] Failed to create session: %s",
             g_ort->GetErrorMessage(status));
    g_ort->ReleaseStatus(status);
    dt_ai_unload_model(ctx);
    return NULL;
  }

  status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                      &ctx->memory_info);
  if (status) {
    g_ort->ReleaseStatus(status);
    dt_ai_unload_model(ctx);
    return NULL;
  }

  // Resolve IO Names
  status = g_ort->GetAllocatorWithDefaultOptions(&ctx->allocator);
  if (status) {
    g_ort->ReleaseStatus(status);
    dt_ai_unload_model(ctx);
    return NULL;
  }

  status = g_ort->SessionGetInputCount(ctx->session, &ctx->input_count);
  if (status) {
    g_ort->ReleaseStatus(status);
    dt_ai_unload_model(ctx);
    return NULL;
  }

  status = g_ort->SessionGetOutputCount(ctx->session, &ctx->output_count);
  if (status) {
    g_ort->ReleaseStatus(status);
    dt_ai_unload_model(ctx);
    return NULL;
  }

  ctx->input_names = g_new0(char *, ctx->input_count);
  ctx->input_types = g_new0(dt_ai_dtype_t, ctx->input_count);
  for (size_t i = 0; i < ctx->input_count; i++) {
    status = g_ort->SessionGetInputName(ctx->session, i, ctx->allocator,
                                        &ctx->input_names[i]);
    if (status) {
      g_ort->ReleaseStatus(status);
      dt_ai_unload_model(ctx);
      return NULL;
    }

    // Get Input Type
    OrtTypeInfo *typeinfo = NULL;
    status = g_ort->SessionGetInputTypeInfo(ctx->session, i, &typeinfo);
    if (status) { // Warning: if this fails, we might still proceed or fail?
                  // Fail safer.
      g_ort->ReleaseStatus(status);
      dt_ai_unload_model(ctx);
      return NULL;
    }
    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;
    status = g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
    if (status) {
      g_ort->ReleaseStatus(status);
      g_ort->ReleaseTypeInfo(typeinfo);
      dt_ai_unload_model(ctx);
      return NULL;
    }
    ONNXTensorElementDataType type;
    status = g_ort->GetTensorElementType(tensor_info, &type);
    if (status) {
      g_ort->ReleaseStatus(status);
      g_ort->ReleaseTypeInfo(typeinfo);
      dt_ai_unload_model(ctx);
      return NULL;
    }

    // Map ONNX type to internal type
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      ctx->input_types[i] = DT_AI_FLOAT;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      ctx->input_types[i] = DT_AI_FLOAT16;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
      ctx->input_types[i] = DT_AI_UINT8;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
      ctx->input_types[i] = DT_AI_INT8;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
      ctx->input_types[i] = DT_AI_INT32;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
      ctx->input_types[i] = DT_AI_INT64;
    else {
      dt_print(DT_DEBUG_AI, "[darktable_ai] Unsupported ONNX input type %d for input %zu", type, i);
      g_ort->ReleaseTypeInfo(typeinfo);
      dt_ai_unload_model(ctx);
      return NULL;
    }

    g_ort->ReleaseTypeInfo(typeinfo);
  }

  ctx->output_names = g_new0(char *, ctx->output_count);
  ctx->output_types = g_new0(dt_ai_dtype_t, ctx->output_count);
  for (size_t i = 0; i < ctx->output_count; i++) {
    status = g_ort->SessionGetOutputName(ctx->session, i, ctx->allocator,
                                         &ctx->output_names[i]);
    if (status) {
      g_ort->ReleaseStatus(status);
      dt_ai_unload_model(ctx);
      return NULL;
    }

    // Get Output Type
    OrtTypeInfo *typeinfo = NULL;
    status = g_ort->SessionGetOutputTypeInfo(ctx->session, i, &typeinfo);
    if (status) {
      g_ort->ReleaseStatus(status);
      dt_ai_unload_model(ctx);
      return NULL;
    }
    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;
    status = g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
    if (status) {
      g_ort->ReleaseStatus(status);
      g_ort->ReleaseTypeInfo(typeinfo);
      dt_ai_unload_model(ctx);
      return NULL;
    }
    ONNXTensorElementDataType type;
    status = g_ort->GetTensorElementType(tensor_info, &type);
    if (status) {
      g_ort->ReleaseStatus(status);
      g_ort->ReleaseTypeInfo(typeinfo);
      dt_ai_unload_model(ctx);
      return NULL;
    }

    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      ctx->output_types[i] = DT_AI_FLOAT;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      ctx->output_types[i] = DT_AI_FLOAT16;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
      ctx->output_types[i] = DT_AI_UINT8;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
      ctx->output_types[i] = DT_AI_INT8;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
      ctx->output_types[i] = DT_AI_INT32;
    else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
      ctx->output_types[i] = DT_AI_INT64;
    else {
      dt_print(DT_DEBUG_AI, "[darktable_ai] Unsupported ONNX output type %d for output %zu", type, i);
      g_ort->ReleaseTypeInfo(typeinfo);
      dt_ai_unload_model(ctx);
      return NULL;
    }

    g_ort->ReleaseTypeInfo(typeinfo);
  }

  return ctx;
}

DT_AI_EXPORT int dt_ai_run(dt_ai_context_t *ctx, dt_ai_tensor_t *inputs,
                           int num_inputs, dt_ai_tensor_t *outputs,
                           int num_outputs) {
  if (!ctx || !ctx->session)
    return -1;
  if (num_inputs != ctx->input_count || num_outputs != ctx->output_count) {
    dt_print(DT_DEBUG_AI,
             "[darktable_ai] IO count mismatch. Expected %zu/%zu, got %d/%d",
             ctx->input_count, ctx->output_count, num_inputs, num_outputs);
    return -2;
  }

  // Run
  OrtStatus *status = NULL;
  int ret = 0;

  // Track temporary buffers to free later
  void **temp_input_buffers = g_new0(void *, num_inputs);

  // Create Input Tensors
  OrtValue **input_tensors = g_new0(OrtValue *, num_inputs);
  OrtValue **output_tensors = g_new0(OrtValue *, num_outputs);
  const char **input_names = (const char **)ctx->input_names; // Cast for Run()

  for (int i = 0; i < num_inputs; i++) {
    int64_t element_count = 1;
    for (int j = 0; j < inputs[i].ndim; j++)
      element_count *= inputs[i].shape[j];

    ONNXTensorElementDataType onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);
    void *data_ptr = inputs[i].data;

    // Check for Type Mismatch (Float -> Half)
    if (inputs[i].type == DT_AI_FLOAT && ctx->input_types[i] == DT_AI_FLOAT16) {
      dt_print(DT_DEBUG_AI, "[darktable_ai] Auto-converting Input[%d] Float32 -> Float16", i);
      // Auto-convert Float32 -> Float16
      onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      type_size = sizeof(uint16_t); // Half is 2 bytes

      uint16_t *half_data = g_malloc(element_count * type_size);
      const float *src = (const float *)inputs[i].data;
      for (int64_t k = 0; k < element_count; k++) {
        half_data[k] = _float_to_half(src[k]);
      }

      data_ptr = half_data;
      temp_input_buffers[i] = half_data;
    } else {
      // Standard Mapping
      switch (inputs[i].type) {
      case DT_AI_FLOAT:
        onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        type_size = sizeof(float);
        break;
      case DT_AI_FLOAT16:
        onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        type_size = sizeof(uint16_t);
        break;
      case DT_AI_UINT8:
        onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        type_size = sizeof(uint8_t);
        break;
      case DT_AI_INT8:
        onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        type_size = sizeof(int8_t);
        break;
      case DT_AI_INT64:
        onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        type_size = sizeof(int64_t);
        break;
      case DT_AI_INT32:
        onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        type_size = sizeof(int32_t);
        break;
      default:
        dt_print(DT_DEBUG_AI, "[darktable_ai] Unsupported input type %d for Input[%d]",
                 inputs[i].type, i);
        ret = -4;
        goto cleanup;
      }
    }

    status = g_ort->CreateTensorWithDataAsOrtValue(
        ctx->memory_info, data_ptr, element_count * type_size, inputs[i].shape,
        inputs[i].ndim, onnx_type, &input_tensors[i]);

    if (status) {
      dt_print(DT_DEBUG_AI, "[darktable_ai] CreateTensor Input[%d] fail: %s", i,
               g_ort->GetErrorMessage(status));
      g_ort->ReleaseStatus(status);
      ret = -4;
      goto cleanup;
    }
  }

  // Create Output Tensors (Pre-allocated)
  const char **output_names = (const char **)ctx->output_names;

  for (int i = 0; i < num_outputs; i++) {
    // Check for Type Mismatch (Float16 -> Float)
    if (outputs[i].type == DT_AI_FLOAT &&
        ctx->output_types[i] == DT_AI_FLOAT16) {
      // Let ORT allocate the output tensor (Float16)
      output_tensors[i] = NULL;
      continue;
    }

    int64_t element_count = 1;
    for (int j = 0; j < outputs[i].ndim; j++)
      element_count *= outputs[i].shape[j];

    ONNXTensorElementDataType onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    switch (outputs[i].type) {
    case DT_AI_FLOAT:
      onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      type_size = sizeof(float);
      break;
    case DT_AI_FLOAT16:
      onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      type_size = sizeof(uint16_t);
      break;
    case DT_AI_UINT8:
      onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      type_size = sizeof(uint8_t);
      break;
    case DT_AI_INT8:
      onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
      type_size = sizeof(int8_t);
      break;
    case DT_AI_INT64:
      onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      type_size = sizeof(int64_t);
      break;
    case DT_AI_INT32:
      onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      type_size = sizeof(int32_t);
      break;
    default:
      dt_print(DT_DEBUG_AI, "[darktable_ai] Unsupported output type %d for Output[%d]",
               outputs[i].type, i);
      ret = -4;
      goto cleanup;
    }

    status = g_ort->CreateTensorWithDataAsOrtValue(
        ctx->memory_info, outputs[i].data, element_count * type_size,
        outputs[i].shape, outputs[i].ndim, onnx_type, &output_tensors[i]);

    if (status) {
      dt_print(DT_DEBUG_AI, "[darktable_ai] CreateTensor Output[%d] fail: %s",
               i, g_ort->GetErrorMessage(status));
      g_ort->ReleaseStatus(status);
      ret = -4;
      goto cleanup;
    }
  }

  // RUN
  status = g_ort->Run(ctx->session, NULL, input_names,
                      (const OrtValue *const *)input_tensors, num_inputs,
                      output_names, num_outputs, output_tensors);

  if (status) {
    dt_print(DT_DEBUG_AI, "[darktable_ai] Run error: %s",
             g_ort->GetErrorMessage(status));
    g_ort->ReleaseStatus(status);
    ret = -3;
  } else {
    // Post-Run: Auto-convert Output (Half -> Float)
    for (int i = 0; i < num_outputs; i++) {
      if (outputs[i].type == DT_AI_FLOAT &&
          ctx->output_types[i] == DT_AI_FLOAT16) {
        if (output_tensors[i]) {
          void *raw_data = NULL;
          status = g_ort->GetTensorMutableData(output_tensors[i], &raw_data);
          if (status) {
            dt_print(DT_DEBUG_AI, "[darktable_ai] GetTensorMutableData failed: %s",
                     g_ort->GetErrorMessage(status));
            g_ort->ReleaseStatus(status);
            continue;
          }

          // Element count
          int64_t element_count = 1;
          for (int j = 0; j < outputs[i].ndim; j++)
            element_count *= outputs[i].shape[j];

          uint16_t *half_data = (uint16_t *)raw_data;
          float *dst = (float *)outputs[i].data;
          for (int64_t k = 0; k < element_count; k++) {
            dst[k] = _half_to_float(half_data[k]);
          }
        }
      }
    }
  }

cleanup:
  // Cleanup OrtValues (Wrappers only, data is owned by caller)
  for (int i = 0; i < num_inputs; i++)
    if (input_tensors[i])
      g_ort->ReleaseValue(input_tensors[i]);
  for (int i = 0; i < num_outputs; i++)
    if (output_tensors[i])
      g_ort->ReleaseValue(output_tensors[i]);

  // Free temp input buffers
  for (int i = 0; i < num_inputs; i++) {
    if (temp_input_buffers[i])
      g_free(temp_input_buffers[i]);
  }
  g_free(temp_input_buffers);

  g_free(input_tensors);
  g_free(output_tensors);

  return ret;
}

DT_AI_EXPORT int dt_ai_get_input_count(dt_ai_context_t *ctx) {
  return ctx ? (int)ctx->input_count : 0;
}

DT_AI_EXPORT int dt_ai_get_output_count(dt_ai_context_t *ctx) {
  return ctx ? (int)ctx->output_count : 0;
}

DT_AI_EXPORT const char *dt_ai_get_input_name(dt_ai_context_t *ctx,
                                               int index) {
  if (!ctx || index < 0 || (size_t)index >= ctx->input_count)
    return NULL;
  return ctx->input_names[index];
}

DT_AI_EXPORT dt_ai_dtype_t dt_ai_get_input_type(dt_ai_context_t *ctx,
                                                  int index) {
  if (!ctx || index < 0 || (size_t)index >= ctx->input_count)
    return DT_AI_FLOAT;
  return ctx->input_types[index];
}

DT_AI_EXPORT const char *dt_ai_get_output_name(dt_ai_context_t *ctx,
                                                int index) {
  if (!ctx || index < 0 || (size_t)index >= ctx->output_count)
    return NULL;
  return ctx->output_names[index];
}

DT_AI_EXPORT dt_ai_dtype_t dt_ai_get_output_type(dt_ai_context_t *ctx,
                                                   int index) {
  if (!ctx || index < 0 || (size_t)index >= ctx->output_count)
    return DT_AI_FLOAT;
  return ctx->output_types[index];
}

DT_AI_EXPORT void dt_ai_unload_model(dt_ai_context_t *ctx) {
  if (ctx) {
    if (ctx->session)
      g_ort->ReleaseSession(ctx->session);
    if (ctx->env)
      g_ort->ReleaseEnv(ctx->env);
    if (ctx->memory_info)
      g_ort->ReleaseMemoryInfo(ctx->memory_info);

    // Release IO names using the allocator that created them
    if (ctx->allocator) {
      for (size_t i = 0; i < ctx->input_count; i++) {
        if (ctx->input_names[i])
          ctx->allocator->Free(ctx->allocator, ctx->input_names[i]);
      }
      for (size_t i = 0; i < ctx->output_count; i++) {
        if (ctx->output_names[i])
          ctx->allocator->Free(ctx->allocator, ctx->output_names[i]);
      }
      // Allocator itself usually doesn't need explicit release if retrieved via
      // GetAllocatorWithDefaultOptions but if we created it, we might. Default
      // options allocator is usually global/shared.
    }

    g_free(ctx->input_names);
    g_free(ctx->output_names);
    g_free(ctx->input_types);
    g_free(ctx->output_types);
    g_free(ctx->loaded_model_id);
    g_free(ctx);
  }
}
