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

#include "server/server.h"
#include "common/colorspaces.h"
#include "common/history.h"
#include "common/image.h"
#include "common/image_cache.h"
#include "common/mipmap_cache.h"
#include "common/iop_order.h"
#include "develop/develop.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/pixelpipe.h"
#include "common/database.h"
#include "common/debug.h"

#include <float.h>
#include <math.h>
#include <string.h>

// Mirror of dt_iop_exposure_params_t from iop/exposure.c
// Must match the struct layout exactly (introspection version 7).
typedef enum _server_exposure_mode_t
{
  _EXPOSURE_MODE_MANUAL = 0,
  _EXPOSURE_MODE_DEFLICKER = 1
} _server_exposure_mode_t;

typedef struct _server_exposure_params_t
{
  _server_exposure_mode_t mode;
  float black;
  float exposure;
  float deflicker_percentile;
  float deflicker_target_level;
  gboolean compensate_exposure_bias;
  gboolean compensate_hilite_pres;
} _server_exposure_params_t;

// Mirror of dt_iop_temperature_params_t from iop/temperature.c
// Must match the struct layout exactly (introspection version 4).
typedef struct _server_temperature_params_t
{
  float red;
  float green;
  float blue;
  float various;
  int preset;
} _server_temperature_params_t;

// --- Runtime verification of mirror struct layouts against IOP introspection ---
// The original IOP param types are defined in .c files (not headers), so we
// can't use _Static_assert. Instead we verify sizeof + field offsets at first
// use via introspection metadata. Mismatches abort immediately.

static void _verify_field(const dt_introspection_type_struct_t *root,
                          const char *struct_name,
                          const char *field_name,
                          size_t expected_offset,
                          size_t expected_size)
{
  for(size_t i = 0; i < root->entries; i++)
  {
    const dt_introspection_field_t *f = root->fields[i];
    if(!f->header.field_name || strcmp(f->header.field_name, field_name)) continue;
    if(f->header.offset != expected_offset)
    {
      fprintf(stderr, "[server] FATAL: %s.%s offset mismatch: IOP=%zu mirror=%zu\n",
              struct_name, field_name, f->header.offset, expected_offset);
      abort();
    }
    if(f->header.size != expected_size)
    {
      fprintf(stderr, "[server] FATAL: %s.%s size mismatch: IOP=%zu mirror=%zu\n",
              struct_name, field_name, f->header.size, expected_size);
      abort();
    }
    return;
  }
  fprintf(stderr, "[server] FATAL: %s.%s not found in introspection\n", struct_name, field_name);
  abort();
}

static void _verify_mirror_structs(void)
{
  static gboolean verified = FALSE;
  if(verified) return;
  verified = TRUE;

  for(GList *l = darktable.iop; l; l = g_list_next(l))
  {
    dt_iop_module_so_t *so = l->data;
    if(!so->get_introspection) continue;
    dt_introspection_t *intro = so->get_introspection();
    if(!intro || !intro->field || intro->field->header.type != DT_INTROSPECTION_TYPE_STRUCT)
      continue;

    const dt_introspection_type_struct_t *root = &intro->field->Struct;

    if(!strcmp(so->op, "exposure"))
    {
      if(intro->size != sizeof(_server_exposure_params_t))
      {
        fprintf(stderr, "[server] FATAL: exposure params size mismatch: IOP=%zu mirror=%zu\n",
                intro->size, sizeof(_server_exposure_params_t));
        abort();
      }
      _verify_field(root, "exposure", "mode",
                    offsetof(_server_exposure_params_t, mode),
                    sizeof(((_server_exposure_params_t *)0)->mode));
      _verify_field(root, "exposure", "exposure",
                    offsetof(_server_exposure_params_t, exposure),
                    sizeof(((_server_exposure_params_t *)0)->exposure));
      _verify_field(root, "exposure", "black",
                    offsetof(_server_exposure_params_t, black),
                    sizeof(((_server_exposure_params_t *)0)->black));
      fprintf(stderr, "[server] exposure mirror struct verified (size=%zu)\n", intro->size);
    }
    else if(!strcmp(so->op, "temperature"))
    {
      if(intro->size != sizeof(_server_temperature_params_t))
      {
        fprintf(stderr, "[server] FATAL: temperature params size mismatch: IOP=%zu mirror=%zu\n",
                intro->size, sizeof(_server_temperature_params_t));
        abort();
      }
      _verify_field(root, "temperature", "red",
                    offsetof(_server_temperature_params_t, red),
                    sizeof(((_server_temperature_params_t *)0)->red));
      _verify_field(root, "temperature", "green",
                    offsetof(_server_temperature_params_t, green),
                    sizeof(((_server_temperature_params_t *)0)->green));
      _verify_field(root, "temperature", "blue",
                    offsetof(_server_temperature_params_t, blue),
                    sizeof(((_server_temperature_params_t *)0)->blue));
      _verify_field(root, "temperature", "preset",
                    offsetof(_server_temperature_params_t, preset),
                    sizeof(((_server_temperature_params_t *)0)->preset));
      fprintf(stderr, "[server] temperature mirror struct verified (size=%zu)\n", intro->size);
    }
  }
}

// --- Spectral conversion copied from iop/temperature.c (exact same logic) ---

#include "external/cie_colorimetric_tables.c"

#define INITIALBLACKBODYTEMPERATURE 4000
#define DT_IOP_LOWEST_TEMPERATURE 1901
#define DT_IOP_HIGHEST_TEMPERATURE 25000

typedef double((*_server_spd)(unsigned long int wavelength, double TempK));

static double _server_spd_blackbody(unsigned long int wavelength, double TempK)
{
  const long double lambda = (double)wavelength * 1e-9;
#define c1 3.7417715246641281639549488324352159753e-16L
#define c2 0.014387769599838156481252937624049081933L
  return (double)(c1 / (powl(lambda, 5) * (expl(c2 / (lambda * TempK)) - 1.0L)));
#undef c2
#undef c1
}

static double _server_spd_daylight(unsigned long int wavelength, double TempK)
{
  cmsCIExyY WhitePoint = { D65xyY.x, D65xyY.y, 1.0 };
  cmsWhitePointFromTemp(&WhitePoint, TempK);

  const double M = (0.0241 + 0.2562 * WhitePoint.x - 0.7341 * WhitePoint.y),
               m1 = (-1.3515 - 1.7703 * WhitePoint.x + 5.9114 * WhitePoint.y) / M,
               m2 = (0.0300 - 31.4424 * WhitePoint.x + 30.0717 * WhitePoint.y) / M;

  const unsigned long int j
      = ((wavelength - cie_daylight_components[0].wavelength)
         / (cie_daylight_components[1].wavelength
            - cie_daylight_components[0].wavelength));

  return (cie_daylight_components[j].S[0] + m1 * cie_daylight_components[j].S[1]
          + m2 * cie_daylight_components[j].S[2]);
}

static cmsCIEXYZ _server_spectrum_to_XYZ(double TempK, _server_spd I)
{
  cmsCIEXYZ Source = {.X = 0.0, .Y = 0.0, .Z = 0.0 };

  for(size_t i = 0; i < cie_1931_std_colorimetric_observer_count; i++)
  {
    const unsigned long int lambda =
      cie_1931_std_colorimetric_observer[0].wavelength
      + (cie_1931_std_colorimetric_observer[1].wavelength
         - cie_1931_std_colorimetric_observer[0].wavelength) * i;

    const double P = I(lambda, TempK);
    Source.X += P * cie_1931_std_colorimetric_observer[i].xyz.X;
    Source.Y += P * cie_1931_std_colorimetric_observer[i].xyz.Y;
    Source.Z += P * cie_1931_std_colorimetric_observer[i].xyz.Z;
  }

  const double _max = fmax(fmax(Source.X, Source.Y), Source.Z);
  if(_max > 0.0)
  {
    Source.X /= _max;
    Source.Y /= _max;
    Source.Z /= _max;
  }
  return Source;
}

static cmsCIEXYZ _server_temperature_to_XYZ(double TempK)
{
  if(TempK < DT_IOP_LOWEST_TEMPERATURE) TempK = DT_IOP_LOWEST_TEMPERATURE;
  if(TempK > DT_IOP_HIGHEST_TEMPERATURE) TempK = DT_IOP_HIGHEST_TEMPERATURE;

  if(TempK < INITIALBLACKBODYTEMPERATURE)
    return _server_spectrum_to_XYZ(TempK, _server_spd_blackbody);
  else
    return _server_spectrum_to_XYZ(TempK, _server_spd_daylight);
}

#define DT_IOP_LOWEST_TINT 0.135
#define DT_IOP_HIGHEST_TINT 2.326

// Binary search inversion: XYZ → temperature + tint (exact copy from temperature.c)
static void _server_XYZ_to_temperature(cmsCIEXYZ XYZ, float *TempK, float *tint)
{
  double maxtemp = DT_IOP_HIGHEST_TEMPERATURE, mintemp = DT_IOP_LOWEST_TEMPERATURE;
  cmsCIEXYZ _xyz;

  for(*TempK = (maxtemp + mintemp) / 2.0;
      (maxtemp - mintemp) > 1.0;
      *TempK = (maxtemp + mintemp) / 2.0)
  {
    _xyz = _server_temperature_to_XYZ(*TempK);
    if(_xyz.Z / _xyz.X > XYZ.Z / XYZ.X)
      maxtemp = *TempK;
    else
      mintemp = *TempK;
  }

  *tint = (_xyz.Y / _xyz.X) / (XYZ.Y / XYZ.X);

  if(*TempK < DT_IOP_LOWEST_TEMPERATURE) *TempK = DT_IOP_LOWEST_TEMPERATURE;
  if(*TempK > DT_IOP_HIGHEST_TEMPERATURE) *TempK = DT_IOP_HIGHEST_TEMPERATURE;
  if(*tint < DT_IOP_LOWEST_TINT) *tint = DT_IOP_LOWEST_TINT;
  if(*tint > DT_IOP_HIGHEST_TINT) *tint = DT_IOP_HIGHEST_TINT;
}

// Convert temperature (Kelvin) + tint to RGB channel multipliers.
// Exact match of temperature.c's _temp2mul + _xyz2mul.
static gboolean _server_temp_tint_to_mul(const dt_image_t *img,
                                          double temp_k, double tint,
                                          double mul[4])
{
  // Get camera color matrices
  float d65_color_matrix[9];
  memcpy(d65_color_matrix, img->d65_color_matrix, sizeof(d65_color_matrix));
  double CAM_to_XYZ[3][4], XYZ_to_CAM[4][3];
  if(!dt_colorspaces_conversion_matrices_xyz(
       img->adobe_XYZ_to_CAM, d65_color_matrix,
       XYZ_to_CAM, CAM_to_XYZ))
    return FALSE;

  // Clamp inputs
  if(temp_k < DT_IOP_LOWEST_TEMPERATURE) temp_k = DT_IOP_LOWEST_TEMPERATURE;
  if(temp_k > DT_IOP_HIGHEST_TEMPERATURE) temp_k = DT_IOP_HIGHEST_TEMPERATURE;
  if(tint < 0.135) tint = 0.135;
  if(tint > 2.326) tint = 2.326;

  // Step 1: Temperature → XYZ via spectral integration (exact darktable method)
  cmsCIEXYZ xyz = _server_temperature_to_XYZ(temp_k);

  // Step 2: Apply tint (same as darktable)
  xyz.Y /= tint;

  // Step 3: XYZ → camera multipliers (same as darktable's _xyz2mul)
  double XYZ[3] = { xyz.X, xyz.Y, xyz.Z };
  double CAM[4];
  for(int k = 0; k < 4; k++)
  {
    CAM[k] = 0.0;
    for(int i = 0; i < 3; i++)
      CAM[k] += XYZ_to_CAM[k][i] * XYZ[i];
  }

  for(int k = 0; k < 4; k++)
    mul[k] = (fabs(CAM[k]) > 1e-10) ? 1.0 / CAM[k] : 0.0;

  // Normalize so green (index 1) = 1
  if(fabs(mul[1]) > 1e-10)
  {
    mul[0] /= mul[1];
    mul[2] /= mul[1];
    mul[3] /= mul[1];
    mul[1] = 1.0;
  }
  else
    return FALSE;

  return TRUE;
}

// Mirror of dt_iop_exposure_data_t — committed pipe data (after process)
typedef struct _server_exposure_data_t
{
  _server_exposure_params_t params;
  gboolean deflicker;
  float black;
  float scale;
  uint32_t *deflicker_histogram;
  dt_dev_histogram_stats_t deflicker_histogram_stats;
} _server_exposure_data_t;

// Lazily allocate SHM buffers for a session (IPC mode only).
// Called from pipeline worker on first render — avoids allocating memory
// until the preview is actually needed.
static gboolean _ensure_shm_allocated(dt_server_t *server, dt_server_session_t *session)
{
  if(server->embedded || session->shm_allocated)
    return TRUE;

  char shm_name0[32], shm_name1[32];
  snprintf(shm_name0, sizeof(shm_name0), "/dt-prev-%s-0", session->session_id);
  snprintf(shm_name1, sizeof(shm_name1), "/dt-prev-%s-1", session->session_id);

  const int w = session->preview_width;
  const int h = session->preview_height;

  if(!dt_shm_create(&session->shm_buffers[0], shm_name0, w, h)
     || !dt_shm_create(&session->shm_buffers[1], shm_name1, w, h))
  {
    dt_shm_destroy(&session->shm_buffers[0]);
    dt_shm_destroy(&session->shm_buffers[1]);
    fprintf(stderr, "[server] SHM allocation failed for session %s\n", session->session_id);
    return FALSE;
  }

  session->shm_allocated = TRUE;
  const size_t total_bytes = (size_t)w * h * 4 * 2;
  fprintf(stderr, "[server] SHM allocated: session=%s %dx%d (%.1f MB)\n",
          session->session_id, w, h, total_bytes / (1024.0 * 1024.0));
  return TRUE;
}

static dt_server_session_t *_find_free_session_slot(dt_server_t *server)
{
  if(server->session_count >= DT_SERVER_MAX_SESSIONS)
    return NULL;

  dt_server_session_t *session = g_new0(dt_server_session_t, 1);
  snprintf(session->session_id, sizeof(session->session_id),
           "dev-%03d", server->next_session_id++);
  server->sessions[server->session_count] = session;
  server->session_count++;
  return session;
}

char *dt_server_develop_open(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "imgid"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing imgid parameter");

  const dt_imgid_t imgid = (dt_imgid_t)json_object_get_int_member(req->params, "imgid");

  // Default preview size
  int preview_width = 1920;
  int preview_height = 1280;
  if(json_object_has_member(req->params, "width"))
    preview_width = (int)json_object_get_int_member(req->params, "width");
  if(json_object_has_member(req->params, "height"))
    preview_height = (int)json_object_get_int_member(req->params, "height");

  // Clamp preview dimensions to screen-friendly caps
  if(preview_width < 320) preview_width = 320;
  if(preview_width > DT_SERVER_MAX_PREVIEW_WIDTH) preview_width = DT_SERVER_MAX_PREVIEW_WIDTH;
  if(preview_height < 240) preview_height = 240;
  if(preview_height > DT_SERVER_MAX_PREVIEW_HEIGHT) preview_height = DT_SERVER_MAX_PREVIEW_HEIGHT;

  // Single-client model: close all existing sessions before opening a new one.
  // This prevents orphaned sessions from accumulating when the client rapidly
  // switches images (the previous close may not have arrived yet).
  for(int i = server->session_count - 1; i >= 0; i--)
  {
    if(server->sessions[i])
    {
      dt_server_session_t *old = server->sessions[i];
      fprintf(stderr, "[server] develop.open: auto-closing session %s\n", old->session_id);
      if(old->shm_allocated)
      {
        dt_shm_destroy(&old->shm_buffers[0]);
        dt_shm_destroy(&old->shm_buffers[1]);
        fprintf(stderr, "[server] SHM freed: session=%s\n", old->session_id);
      }
      pthread_mutex_destroy(&old->pipeline_mutex);
      dt_dev_cleanup(&old->dev);
      g_free(old);
      server->sessions[i] = NULL;
    }
  }
  server->session_count = 0;

  // Allocate session slot
  dt_server_session_t *session = _find_free_session_slot(server);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_BUSY,
                                 "Maximum number of develop sessions reached");

  session->imgid = imgid;
  session->preview_width = preview_width;
  session->preview_height = preview_height;
  session->frame_sequence = 0;
  session->dirty = TRUE;
  session->pipeline_seq = 0;
  session->pipeline_busy = FALSE;
  pthread_mutex_init(&session->pipeline_mutex, NULL);

  // Initialize develop context following dt_dev_image() pattern (develop.c:3847)
  dt_dev_init(&session->dev, TRUE);
  session->dev.gui_attached = FALSE;

  // The full pipe is already DT_DEV_PIXELPIPE_FULL from dt_dev_pixelpipe_init().
  // Do NOT set DT_DEV_PIXELPIPE_IMAGE — that flag disables the intermediate
  // cache (pipe->nocache=TRUE), which prevents reuse of upstream module outputs
  // when only one module's params change.
  dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;

  // Load image: instantiates modules, loads history from DB
  dt_dev_load_image(&session->dev, imgid);

  // Apply history to module structs (params + enabled state).
  // dt_dev_load_image() reads history items but does not sync them to modules.
  // Without this, module->enabled stays at default_enabled and module->params
  // stay at default_params, causing set_params to create disabled history items.
  dt_pthread_mutex_lock(&session->dev.history_mutex);
  dt_dev_pop_history_items_ext(&session->dev, session->dev.history_end);
  dt_pthread_mutex_unlock(&session->dev.history_mutex);

  // Validate the image was loaded successfully
  if(!dt_is_valid_imgid(session->dev.image_storage.id))
  {
    dt_dev_cleanup(&session->dev);
    server->sessions[server->session_count - 1] = NULL;
    server->session_count--;
    g_free(session);
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND,
                                 "Failed to load image into develop");
  }

  // Configure viewport for requested preview dimensions
  session->dev.full.zoom = DT_ZOOM_FIT;
  session->dev.full.width = preview_width;
  session->dev.full.height = preview_height;
  session->dev.full.ppd = 1.0;
  session->dev.full.color_assessment = FALSE;
  session->dev.full.dev = &session->dev;
  session->dev.full.pipe = pipe;

  // SHM buffers are allocated lazily on first pipeline render (_ensure_shm_allocated).
  // This saves memory when sessions are opened but preview is not immediately needed.
  session->shm_allocated = FALSE;
  session->front_buffer = 0;

  // Pre-compute SHM names for the response (client maps them after first render)
  char shm_name0[32] = {0}, shm_name1[32] = {0};
  if(!server->embedded)
  {
    snprintf(shm_name0, sizeof(shm_name0), "/dt-prev-%s-0", session->session_id);
    snprintf(shm_name1, sizeof(shm_name1), "/dt-prev-%s-1", session->session_id);
  }

  fprintf(stderr, "[server] develop.open: session=%s imgid=%d preview=%dx%d embedded=%d (SHM deferred)\n",
          session->session_id, imgid, preview_width, preview_height, server->embedded);

  // Build response
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "session_id");
  json_builder_add_string_value(b, session->session_id);

  json_builder_set_member_name(b, "imgid");
  json_builder_add_int_value(b, imgid);

  json_builder_set_member_name(b, "preview_width");
  json_builder_add_int_value(b, preview_width);

  json_builder_set_member_name(b, "preview_height");
  json_builder_add_int_value(b, preview_height);

  json_builder_set_member_name(b, "shm_names");
  json_builder_begin_array(b);
  if(!server->embedded)
  {
    json_builder_add_string_value(b, shm_name0);
    json_builder_add_string_value(b, shm_name1);
  }
  json_builder_end_array(b);

  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_close(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  fprintf(stderr, "[server] develop.close: session=%s dirty=%d\n", session_id, session->dirty);

  // Write in-memory history to DB before closing
  if(session->dirty)
  {
    dt_dev_write_history(&session->dev);

    // Invalidate mipmap cache so thumbnails regenerate with new edits
    // (mirrors GTK darkroom.c leave behavior)
    const dt_imgid_t imgid = session->imgid;
    dt_mipmap_cache_remove(imgid);
    dt_image_update_final_size(imgid);
    dt_image_synch_xmp(imgid);
  }

  // Destroy SHM buffers if they were allocated
  if(session->shm_allocated)
  {
    dt_shm_destroy(&session->shm_buffers[0]);
    dt_shm_destroy(&session->shm_buffers[1]);
    fprintf(stderr, "[server] SHM freed: session=%s\n", session->session_id);
  }

  // Cleanup pipeline mutex
  pthread_mutex_destroy(&session->pipeline_mutex);

  // Cleanup develop context
  dt_dev_cleanup(&session->dev);

  // Remove from server's session list
  for(int i = 0; i < server->session_count; i++)
  {
    if(server->sessions[i] == session)
    {
      // Shift remaining sessions down
      for(int j = i; j < server->session_count - 1; j++)
        server->sessions[j] = server->sessions[j + 1];
      server->sessions[server->session_count - 1] = NULL;
      server->session_count--;
      break;
    }
  }
  g_free(session);

  // Build response
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "closed");
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_get_modules(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "modules");
  json_builder_begin_array(b);

  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;

    json_builder_begin_object(b);

    json_builder_set_member_name(b, "op");
    json_builder_add_string_value(b, mod->op);

    json_builder_set_member_name(b, "name");
    json_builder_add_string_value(b, mod->name());

    json_builder_set_member_name(b, "enabled");
    json_builder_add_boolean_value(b, mod->enabled);

    json_builder_set_member_name(b, "instance");
    json_builder_add_int_value(b, mod->multi_priority);

    json_builder_set_member_name(b, "multi_name");
    {
      const char *mname = mod->multi_name;
      // Skip _builtin_ prefix
      if(g_str_has_prefix(mname, "_builtin_"))
        mname += strlen("_builtin_");
      // Skip default "0" name
      if(mname[0] && strcmp(mname, "0") != 0)
        json_builder_add_string_value(b, mname);
      else
        json_builder_add_string_value(b, "");
    }

    json_builder_set_member_name(b, "flags");
    json_builder_add_int_value(b, mod->flags());

    json_builder_set_member_name(b, "iop_order");
    json_builder_add_double_value(b, mod->iop_order);

    json_builder_set_member_name(b, "params_size");
    json_builder_add_int_value(b, mod->params_size);

    // Module description from description() callback
    const char **des = mod->description ? mod->description(mod) : NULL;
    if(des && des[0])
    {
      json_builder_set_member_name(b, "description");
      json_builder_begin_object(b);

      json_builder_set_member_name(b, "main");
      json_builder_add_string_value(b, des[0] ? des[0] : "");
      json_builder_set_member_name(b, "purpose");
      json_builder_add_string_value(b, des[1] ? des[1] : "");
      json_builder_set_member_name(b, "input");
      json_builder_add_string_value(b, des[2] ? des[2] : "");
      json_builder_set_member_name(b, "process");
      json_builder_add_string_value(b, des[3] ? des[3] : "");
      json_builder_set_member_name(b, "output");
      json_builder_add_string_value(b, des[4] ? des[4] : "");

      json_builder_end_object(b);
    }

    // Blend parameters (only for modules that support blending)
    if((mod->flags() & IOP_FLAGS_SUPPORTS_BLENDING) && mod->blend_params)
    {
      json_builder_set_member_name(b, "blend");
      json_builder_begin_object(b);

      json_builder_set_member_name(b, "mask_mode");
      json_builder_add_int_value(b, mod->blend_params->mask_mode);

      json_builder_set_member_name(b, "blend_mode");
      json_builder_add_int_value(b, mod->blend_params->blend_mode);

      json_builder_set_member_name(b, "opacity");
      json_builder_add_double_value(b, mod->blend_params->opacity);

      json_builder_set_member_name(b, "mask_id");
      json_builder_add_int_value(b, mod->blend_params->mask_id);

      json_builder_set_member_name(b, "mask_combine");
      json_builder_add_int_value(b, mod->blend_params->mask_combine);

      json_builder_set_member_name(b, "blend_parameter");
      json_builder_add_double_value(b, mod->blend_params->blend_parameter);

      json_builder_set_member_name(b, "details");
      json_builder_add_double_value(b, mod->blend_params->details);

      json_builder_set_member_name(b, "feathering_guide");
      json_builder_add_int_value(b, mod->blend_params->feathering_guide);

      json_builder_set_member_name(b, "feathering_radius");
      json_builder_add_double_value(b, mod->blend_params->feathering_radius);

      json_builder_set_member_name(b, "blur_radius");
      json_builder_add_double_value(b, mod->blend_params->blur_radius);

      json_builder_set_member_name(b, "contrast");
      json_builder_add_double_value(b, mod->blend_params->contrast);

      json_builder_set_member_name(b, "brightness");
      json_builder_add_double_value(b, mod->blend_params->brightness);

      json_builder_end_object(b);
    }

    json_builder_end_object(b);
  }

  json_builder_end_array(b);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_get_history(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "history_end");
  json_builder_add_int_value(b, session->dev.history_end);

  json_builder_set_member_name(b, "items");
  json_builder_begin_array(b);

  // Add "original" as item -1 (same as GTK darktable)
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "num");
  json_builder_add_int_value(b, -1);
  json_builder_set_member_name(b, "history_index");
  json_builder_add_int_value(b, -1);
  json_builder_set_member_name(b, "op");
  json_builder_add_string_value(b, "");
  json_builder_set_member_name(b, "name");
  json_builder_add_string_value(b, "original");
  json_builder_set_member_name(b, "enabled");
  json_builder_add_boolean_value(b, FALSE);
  json_builder_set_member_name(b, "mandatory");
  json_builder_add_boolean_value(b, TRUE);
  json_builder_end_object(b);

  dt_pthread_mutex_lock(&session->dev.history_mutex);

  int num = 0;
  int real_index = 0;
  for(GList *hist = session->dev.history; hist; hist = g_list_next(hist), real_index++)
  {
    dt_dev_history_item_t *item = hist->data;
    if(!item) continue;

    // Skip mask_manager entries
    if(!strcmp(item->op_name, "mask_manager")) continue;

    json_builder_begin_object(b);

    json_builder_set_member_name(b, "num");
    json_builder_add_int_value(b, num++);

    json_builder_set_member_name(b, "history_index");
    json_builder_add_int_value(b, real_index);

    json_builder_set_member_name(b, "op");
    json_builder_add_string_value(b, item->op_name);

    json_builder_set_member_name(b, "name");
    if(item->module && item->module->name)
    {
      const char *display_name = item->module->name();
      // Skip _builtin_ prefix and empty/default multi_name values
      const char *mname = item->multi_name;
      if(g_str_has_prefix(mname, BUILTIN_PREFIX))
        mname += strlen(BUILTIN_PREFIX);
      if(mname[0] && strcmp(mname, "0") != 0)
      {
        char full_name[256];
        snprintf(full_name, sizeof(full_name), "%s \xe2\x80\xa2 %s",
                 display_name, mname);
        json_builder_add_string_value(b, full_name);
      }
      else
        json_builder_add_string_value(b, display_name);
    }
    else
      json_builder_add_string_value(b, item->op_name);

    json_builder_set_member_name(b, "enabled");
    json_builder_add_boolean_value(b, item->enabled);

    json_builder_set_member_name(b, "mandatory");
    json_builder_add_boolean_value(b, item->module && item->module->hide_enable_button);

    json_builder_end_object(b);
  }

  dt_pthread_mutex_unlock(&session->dev.history_mutex);

  json_builder_end_array(b);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

// --- Generic introspection-based param serialization ---

/** Serialize a single introspection field value from params blob to JSON */
static void _introspection_serialize_field(JsonBuilder *b, const dt_introspection_field_t *field,
                                           const void *params)
{
  const void *ptr = (const char *)params + field->header.offset;

  switch(field->header.type)
  {
    case DT_INTROSPECTION_TYPE_FLOAT:
      json_builder_add_double_value(b, *(const float *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_DOUBLE:
      json_builder_add_double_value(b, *(const double *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_INT:
      json_builder_add_int_value(b, *(const int *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_UINT:
      json_builder_add_int_value(b, *(const unsigned int *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_SHORT:
      json_builder_add_int_value(b, *(const short *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_USHORT:
      json_builder_add_int_value(b, *(const unsigned short *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_INT8:
      json_builder_add_int_value(b, *(const int8_t *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_UINT8:
      json_builder_add_int_value(b, *(const uint8_t *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_LONG:
      json_builder_add_int_value(b, *(const long *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_ULONG:
      json_builder_add_int_value(b, (gint64)*(const unsigned long *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_BOOL:
      json_builder_add_boolean_value(b, *(const gboolean *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_ENUM:
      json_builder_add_int_value(b, *(const int *)ptr);
      break;
    case DT_INTROSPECTION_TYPE_CHAR:
    {
      // A single char field — treat as int
      json_builder_add_int_value(b, *(const char *)ptr);
      break;
    }
    case DT_INTROSPECTION_TYPE_ARRAY:
    {
      const dt_introspection_type_array_t *arr = &field->Array;
      // char arrays → JSON string shorthand
      if(arr->field->header.type == DT_INTROSPECTION_TYPE_CHAR)
      {
        json_builder_add_string_value(b, (const char *)ptr);
        break;
      }
      json_builder_begin_array(b);
      for(size_t i = 0; i < arr->count; i++)
      {
        // Create a temporary field descriptor with adjusted offset for each element
        dt_introspection_field_t elem = *arr->field;
        elem.header.offset = field->header.offset + i * arr->field->header.size;
        _introspection_serialize_field(b, &elem, params);
      }
      json_builder_end_array(b);
      break;
    }
    case DT_INTROSPECTION_TYPE_STRUCT:
    {
      json_builder_begin_object(b);
      const dt_introspection_type_struct_t *s = &field->Struct;
      for(size_t i = 0; i < s->entries; i++)
      {
        const dt_introspection_field_t *child = s->fields[i];
        if(!child->header.field_name) continue;
        json_builder_set_member_name(b, child->header.field_name);
        _introspection_serialize_field(b, child, params);
      }
      json_builder_end_object(b);
      break;
    }
    default:
      // OPAQUE, FLOATCOMPLEX, UNION — skip with null
      json_builder_add_null_value(b);
      break;
  }
}

/** Serialize all top-level params fields using introspection */
static gboolean _introspection_serialize_params(JsonBuilder *b, const dt_iop_module_t *module)
{
  dt_introspection_t *intro = module->so->get_introspection();
  if(!intro || !intro->field || intro->field->header.type != DT_INTROSPECTION_TYPE_STRUCT)
    return FALSE;

  const dt_introspection_type_struct_t *root = &intro->field->Struct;
  for(size_t i = 0; i < root->entries; i++)
  {
    const dt_introspection_field_t *child = root->fields[i];
    if(!child->header.field_name) continue;
    json_builder_set_member_name(b, child->header.field_name);
    _introspection_serialize_field(b, child, module->params);
  }
  return TRUE;
}

/** Deserialize a single field from JSON into the params blob */
static void _introspection_deserialize_field(const dt_introspection_field_t *field,
                                             void *params, JsonNode *node)
{
  void *ptr = (char *)params + field->header.offset;

  switch(field->header.type)
  {
    case DT_INTROSPECTION_TYPE_FLOAT:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(float *)ptr = (float)json_node_get_double(node);
      break;
    case DT_INTROSPECTION_TYPE_DOUBLE:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(double *)ptr = json_node_get_double(node);
      break;
    case DT_INTROSPECTION_TYPE_INT:
    case DT_INTROSPECTION_TYPE_ENUM:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(int *)ptr = (int)json_node_get_int(node);
      break;
    case DT_INTROSPECTION_TYPE_UINT:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(unsigned int *)ptr = (unsigned int)json_node_get_int(node);
      break;
    case DT_INTROSPECTION_TYPE_SHORT:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(short *)ptr = (short)json_node_get_int(node);
      break;
    case DT_INTROSPECTION_TYPE_USHORT:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(unsigned short *)ptr = (unsigned short)json_node_get_int(node);
      break;
    case DT_INTROSPECTION_TYPE_INT8:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(int8_t *)ptr = (int8_t)json_node_get_int(node);
      break;
    case DT_INTROSPECTION_TYPE_UINT8:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(uint8_t *)ptr = (uint8_t)json_node_get_int(node);
      break;
    case DT_INTROSPECTION_TYPE_LONG:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(long *)ptr = (long)json_node_get_int(node);
      break;
    case DT_INTROSPECTION_TYPE_ULONG:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(unsigned long *)ptr = (unsigned long)json_node_get_int(node);
      break;
    case DT_INTROSPECTION_TYPE_BOOL:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(gboolean *)ptr = json_node_get_boolean(node);
      break;
    case DT_INTROSPECTION_TYPE_CHAR:
      if(JSON_NODE_HOLDS_VALUE(node))
        *(char *)ptr = (char)json_node_get_int(node);
      break;
    case DT_INTROSPECTION_TYPE_ARRAY:
    {
      const dt_introspection_type_array_t *a = &field->Array;
      // String → char array shorthand
      if(a->field->header.type == DT_INTROSPECTION_TYPE_CHAR && JSON_NODE_HOLDS_VALUE(node))
      {
        const char *s = json_node_get_string(node);
        if(s) g_strlcpy((char *)ptr, s, a->count);
        break;
      }
      if(!JSON_NODE_HOLDS_ARRAY(node)) break;
      JsonArray *arr = json_node_get_array(node);
      const guint len = MIN(json_array_get_length(arr), (guint)a->count);
      for(guint i = 0; i < len; i++)
      {
        dt_introspection_field_t elem = *a->field;
        elem.header.offset = field->header.offset + i * a->field->header.size;
        _introspection_deserialize_field(&elem, params, json_array_get_element(arr, i));
      }
      break;
    }
    case DT_INTROSPECTION_TYPE_STRUCT:
    {
      if(!JSON_NODE_HOLDS_OBJECT(node)) break;
      JsonObject *obj = json_node_get_object(node);
      const dt_introspection_type_struct_t *s = &field->Struct;
      for(size_t i = 0; i < s->entries; i++)
      {
        const dt_introspection_field_t *child = s->fields[i];
        if(!child->header.field_name) continue;
        if(json_object_has_member(obj, child->header.field_name))
          _introspection_deserialize_field(child, params,
            json_object_get_member(obj, child->header.field_name));
      }
      break;
    }
    default:
      break;
  }
}

/** Deserialize JSON params object into module params blob using introspection.
 *  Only fields present in the JSON are modified (partial update). */
static gboolean _introspection_deserialize_params(const dt_iop_module_t *module,
                                                   JsonObject *new_params)
{
  dt_introspection_t *intro = module->so->get_introspection();
  if(!intro || !intro->field || intro->field->header.type != DT_INTROSPECTION_TYPE_STRUCT)
    return FALSE;

  const dt_introspection_type_struct_t *root = &intro->field->Struct;
  for(size_t i = 0; i < root->entries; i++)
  {
    const dt_introspection_field_t *child = root->fields[i];
    if(!child->header.field_name) continue;
    if(!json_object_has_member(new_params, child->header.field_name)) continue;
    JsonNode *node = json_object_get_member(new_params, child->header.field_name);
    _introspection_deserialize_field(child, module->params, node);
  }
  return TRUE;
}

/** Serialize introspection schema for a module (field types, min/max/default, enums) */
static void _introspection_serialize_schema_field(JsonBuilder *b, const dt_introspection_field_t *field)
{
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "name");
  json_builder_add_string_value(b, field->header.field_name ? field->header.field_name : "");

  json_builder_set_member_name(b, "type");
  switch(field->header.type)
  {
    case DT_INTROSPECTION_TYPE_FLOAT:
      json_builder_add_string_value(b, "float");
      json_builder_set_member_name(b, "min");
      json_builder_add_double_value(b, field->Float.Min);
      json_builder_set_member_name(b, "max");
      json_builder_add_double_value(b, field->Float.Max);
      json_builder_set_member_name(b, "default");
      json_builder_add_double_value(b, field->Float.Default);
      break;
    case DT_INTROSPECTION_TYPE_DOUBLE:
      json_builder_add_string_value(b, "double");
      json_builder_set_member_name(b, "min");
      json_builder_add_double_value(b, field->Double.Min);
      json_builder_set_member_name(b, "max");
      json_builder_add_double_value(b, field->Double.Max);
      json_builder_set_member_name(b, "default");
      json_builder_add_double_value(b, field->Double.Default);
      break;
    case DT_INTROSPECTION_TYPE_INT:
      json_builder_add_string_value(b, "int");
      json_builder_set_member_name(b, "min");
      json_builder_add_int_value(b, field->Int.Min);
      json_builder_set_member_name(b, "max");
      json_builder_add_int_value(b, field->Int.Max);
      json_builder_set_member_name(b, "default");
      json_builder_add_int_value(b, field->Int.Default);
      break;
    case DT_INTROSPECTION_TYPE_UINT:
      json_builder_add_string_value(b, "uint");
      json_builder_set_member_name(b, "min");
      json_builder_add_int_value(b, field->UInt.Min);
      json_builder_set_member_name(b, "max");
      json_builder_add_int_value(b, field->UInt.Max);
      json_builder_set_member_name(b, "default");
      json_builder_add_int_value(b, field->UInt.Default);
      break;
    case DT_INTROSPECTION_TYPE_BOOL:
      json_builder_add_string_value(b, "bool");
      json_builder_set_member_name(b, "default");
      json_builder_add_boolean_value(b, field->Bool.Default);
      break;
    case DT_INTROSPECTION_TYPE_ENUM:
      json_builder_add_string_value(b, "enum");
      json_builder_set_member_name(b, "default");
      json_builder_add_int_value(b, field->Enum.Default);
      json_builder_set_member_name(b, "values");
      json_builder_begin_array(b);
      for(size_t j = 0; j < field->Enum.entries; j++)
      {
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "name");
        json_builder_add_string_value(b, field->Enum.values[j].name);
        json_builder_set_member_name(b, "value");
        json_builder_add_int_value(b, field->Enum.values[j].value);
        if(field->Enum.values[j].description)
        {
          json_builder_set_member_name(b, "description");
          json_builder_add_string_value(b, field->Enum.values[j].description);
        }
        json_builder_end_object(b);
      }
      json_builder_end_array(b);
      break;
    case DT_INTROSPECTION_TYPE_SHORT:
      json_builder_add_string_value(b, "short");
      json_builder_set_member_name(b, "min");
      json_builder_add_int_value(b, field->Short.Min);
      json_builder_set_member_name(b, "max");
      json_builder_add_int_value(b, field->Short.Max);
      json_builder_set_member_name(b, "default");
      json_builder_add_int_value(b, field->Short.Default);
      break;
    case DT_INTROSPECTION_TYPE_USHORT:
      json_builder_add_string_value(b, "ushort");
      json_builder_set_member_name(b, "min");
      json_builder_add_int_value(b, field->UShort.Min);
      json_builder_set_member_name(b, "max");
      json_builder_add_int_value(b, field->UShort.Max);
      json_builder_set_member_name(b, "default");
      json_builder_add_int_value(b, field->UShort.Default);
      break;
    case DT_INTROSPECTION_TYPE_INT8:
      json_builder_add_string_value(b, "int8");
      json_builder_set_member_name(b, "min");
      json_builder_add_int_value(b, field->Int8.Min);
      json_builder_set_member_name(b, "max");
      json_builder_add_int_value(b, field->Int8.Max);
      json_builder_set_member_name(b, "default");
      json_builder_add_int_value(b, field->Int8.Default);
      break;
    case DT_INTROSPECTION_TYPE_UINT8:
      json_builder_add_string_value(b, "uint8");
      json_builder_set_member_name(b, "min");
      json_builder_add_int_value(b, field->UInt8.Min);
      json_builder_set_member_name(b, "max");
      json_builder_add_int_value(b, field->UInt8.Max);
      json_builder_set_member_name(b, "default");
      json_builder_add_int_value(b, field->UInt8.Default);
      break;
    case DT_INTROSPECTION_TYPE_ARRAY:
    {
      json_builder_add_string_value(b, "array");
      json_builder_set_member_name(b, "count");
      json_builder_add_int_value(b, field->Array.count);
      json_builder_set_member_name(b, "element");
      _introspection_serialize_schema_field(b, field->Array.field);
      break;
    }
    case DT_INTROSPECTION_TYPE_STRUCT:
    {
      json_builder_add_string_value(b, "struct");
      json_builder_set_member_name(b, "fields");
      json_builder_begin_array(b);
      for(size_t j = 0; j < field->Struct.entries; j++)
        _introspection_serialize_schema_field(b, field->Struct.fields[j]);
      json_builder_end_array(b);
      break;
    }
    default:
      json_builder_add_string_value(b, "opaque");
      break;
  }

  if(field->header.description)
  {
    json_builder_set_member_name(b, "description");
    json_builder_add_string_value(b, field->header.description);
  }

  json_builder_end_object(b);
}

// Exposure applied by the last deflicker run. Same computation as
// _compute_deflicker_correction() in iop/exposure.c, using the raw histogram
// the module caches in its pipe data.
static float _deflicker_applied_exposure(const dt_dev_pixelpipe_t *pipe,
                                         const _server_exposure_data_t *ed)
{
  const _server_exposure_params_t *p = &ed->params;
  const uint32_t *histogram = ed->deflicker_histogram;
  const dt_dev_histogram_stats_t *stats = &ed->deflicker_histogram_stats;

  // without a histogram the module keeps the manual exposure
  if(!histogram) return p->exposure;

  const double thr = CLAMP((double)stats->pixels * (double)p->deflicker_percentile / 100.0,
                           0.0, (double)stats->pixels);
  size_t n = 0;
  uint32_t raw = 0;
  for(size_t i = 0; i < stats->bins_count; i++)
  {
    n += histogram[i];
    if((double)n >= thr)
    {
      raw = i;
      break;
    }
  }

  const uint32_t black_level = (uint32_t)pipe->dsc.rawprepare.raw_black_level;
  const uint32_t raw_max = pipe->dsc.rawprepare.raw_white_point - black_level;
  const int64_t raw_val = MAX((int64_t)raw - (int64_t)black_level, 1);
  const double ev = -log2(raw_max) + log2(raw_val);

  return p->deflicker_target_level - ev;
}

char *dt_server_develop_get_params(dt_server_t *server, const dt_server_request_t *req)
{
  _verify_mirror_structs();

  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or op parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  if(!session_id || !op)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id and op must be strings");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Find the module
  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "op");
  json_builder_add_string_value(b, target->op);

  json_builder_set_member_name(b, "enabled");
  json_builder_add_boolean_value(b, target->enabled);

  // Module-specific parameter serialization
  // For now, we hardcode serializers for known modules.
  json_builder_set_member_name(b, "params");
  json_builder_begin_object(b);

  if(!strcmp(op, "exposure"))
  {
    // Use introspection for base params
    _introspection_serialize_params(b, target);

    // Computed EXIF bias values for dynamic checkbox labels
    float exposure_bias = 0.0f;
    if(session->dev.image_storage.exif_exposure_bias != DT_EXIF_TAG_UNINITIALIZED)
      exposure_bias = CLAMPF(session->dev.image_storage.exif_exposure_bias, -5.0f, 5.0f);
    json_builder_set_member_name(b, "exposure_bias_ev");
    json_builder_add_double_value(b, exposure_bias);

    float highlight_bias = 0.0f;
    if(session->dev.image_storage.exif_highlight_preservation > 0.0f
       && session->dev.image_storage.exif_highlight_preservation != DT_EXIF_TAG_UNINITIALIZED)
      highlight_bias = CLAMPF(session->dev.image_storage.exif_highlight_preservation, -1.0f, 4.0f);
    json_builder_set_member_name(b, "highlight_bias_ev");
    json_builder_add_double_value(b, highlight_bias);

    // Deflicker computed exposure from last pipeline run. Pipe nodes and their
    // data are only safe to walk under busy_mutex; trylock like
    // preview_data.c does, a busy pipe just skips the readout this time.
    dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
    const _server_exposure_params_t *ep = (const _server_exposure_params_t *)target->params;
    if(ep->mode == _EXPOSURE_MODE_DEFLICKER && !dt_pthread_mutex_trylock(&pipe->busy_mutex))
    {
      for(GList *nodes = pipe->nodes; nodes; nodes = g_list_next(nodes))
      {
        dt_dev_pixelpipe_iop_t *piece = nodes->data;
        if(piece->module == target && piece->data)
        {
          const _server_exposure_data_t *ed = (const _server_exposure_data_t *)piece->data;
          if(ed->deflicker)
          {
            json_builder_set_member_name(b, "deflicker_computed_exposure");
            json_builder_add_double_value(b, _deflicker_applied_exposure(pipe, ed));
          }
          break;
        }
      }
      dt_pthread_mutex_unlock(&pipe->busy_mutex);
    }
  }
  else if(!strcmp(op, "temperature"))
  {
    // Use introspection for base params (red, green, blue, various, preset)
    _introspection_serialize_params(b, target);

    // Compute temperature and tint from coefficients using camera color matrix
    const _server_temperature_params_t *p = (const _server_temperature_params_t *)target->params;
    {
      float d65_cm[9];
      memcpy(d65_cm, session->dev.image_storage.d65_color_matrix, sizeof(d65_cm));
      double CAM_to_XYZ[3][4], XYZ_to_CAM[4][3];
      if(dt_colorspaces_conversion_matrices_xyz(
          session->dev.image_storage.adobe_XYZ_to_CAM, d65_cm,
          XYZ_to_CAM, CAM_to_XYZ))
      {
        double CAM[4] = {
          p->red > 0.0f ? 1.0 / p->red : 0.0,
          p->green > 0.0f ? 1.0 / p->green : 0.0,
          p->blue > 0.0f ? 1.0 / p->blue : 0.0,
          p->various > 0.0f ? 1.0 / p->various : 0.0
        };
        double XYZ[3] = { 0, 0, 0 };
        for(int k = 0; k < 3; k++)
          for(int i = 0; i < 4; i++)
            XYZ[k] += CAM_to_XYZ[k][i] * CAM[i];

        if(XYZ[0] > 0 && XYZ[1] > 0 && XYZ[2] > 0)
        {
          cmsCIEXYZ cmsXYZ = { XYZ[0], XYZ[1], XYZ[2] };
          float temp_k, tint;
          _server_XYZ_to_temperature(cmsXYZ, &temp_k, &tint);

          json_builder_set_member_name(b, "temperature_k");
          json_builder_add_double_value(b, temp_k);
          json_builder_set_member_name(b, "tint");
          json_builder_add_double_value(b, tint);
        }
      }
    }
  }
  else if(!strcmp(op, "colorin"))
  {
    // Use introspection for base params (no mirror struct needed)
    _introspection_serialize_params(b, target);

    // Look up fields via introspection for computed values
    dt_introspection_t *intro = target->so->get_introspection();
    const int *ci_type = dt_introspection_get_child(intro->field, target->params, "type", NULL);
    const char *ci_filename = dt_introspection_get_child(intro->field, target->params, "filename", NULL);
    const int *ci_type_work = dt_introspection_get_child(intro->field, target->params, "type_work", NULL);
    const char *ci_filename_work = dt_introspection_get_child(intro->field, target->params, "filename_work", NULL);

    // Computed: current profile display names
    json_builder_set_member_name(b, "input_profile_name");
    json_builder_add_string_value(b, dt_colorspaces_get_name(ci_type ? *ci_type : 0,
                                                              ci_filename ? ci_filename : ""));
    json_builder_set_member_name(b, "work_profile_name");
    json_builder_add_string_value(b, dt_colorspaces_get_name(ci_type_work ? *ci_type_work : 0,
                                                              ci_filename_work ? ci_filename_work : ""));

    // Computed: available input profiles list
    json_builder_set_member_name(b, "input_profiles");
    json_builder_begin_array(b);
    for(GList *l = darktable.color_profiles->profiles; l; l = g_list_next(l))
    {
      dt_colorspaces_color_profile_t *prof = l->data;
      if(prof->in_pos > -1)
      {
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "type");
        json_builder_add_int_value(b, (int)prof->type);
        json_builder_set_member_name(b, "name");
        json_builder_add_string_value(b, prof->name);
        json_builder_set_member_name(b, "filename");
        json_builder_add_string_value(b, prof->filename);
        json_builder_end_object(b);
      }
    }
    json_builder_end_array(b);

    // Computed: available working profiles list
    json_builder_set_member_name(b, "work_profiles");
    json_builder_begin_array(b);
    for(GList *l = darktable.color_profiles->profiles; l; l = g_list_next(l))
    {
      dt_colorspaces_color_profile_t *prof = l->data;
      if(prof->work_pos > -1)
      {
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "type");
        json_builder_add_int_value(b, (int)prof->type);
        json_builder_set_member_name(b, "name");
        json_builder_add_string_value(b, prof->name);
        json_builder_set_member_name(b, "filename");
        json_builder_add_string_value(b, prof->filename);
        json_builder_end_object(b);
      }
    }
    json_builder_end_array(b);
  }
  else if(!strcmp(op, "colorout"))
  {
    // Use introspection for base params (no mirror struct needed)
    _introspection_serialize_params(b, target);

    // Look up fields via introspection for computed values
    dt_introspection_t *intro_co = target->so->get_introspection();
    const int *co_type = dt_introspection_get_child(intro_co->field, target->params, "type", NULL);
    const char *co_filename = dt_introspection_get_child(intro_co->field, target->params, "filename", NULL);

    // Computed: current profile display name
    json_builder_set_member_name(b, "output_profile_name");
    json_builder_add_string_value(b, dt_colorspaces_get_name(co_type ? *co_type : 0,
                                                              co_filename ? co_filename : ""));

    // Computed: available output profiles
    json_builder_set_member_name(b, "output_profiles");
    json_builder_begin_array(b);
    for(GList *l = darktable.color_profiles->profiles; l; l = g_list_next(l))
    {
      dt_colorspaces_color_profile_t *prof = l->data;
      if(prof->out_pos > -1)
      {
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "type");
        json_builder_add_int_value(b, (int)prof->type);
        json_builder_set_member_name(b, "name");
        json_builder_add_string_value(b, prof->name);
        json_builder_set_member_name(b, "filename");
        json_builder_add_string_value(b, prof->filename);
        json_builder_end_object(b);
      }
    }
    json_builder_end_array(b);
  }
  else if(!strcmp(op, "demosaic"))
  {
    // Use introspection for base params
    _introspection_serialize_params(b, target);

    // Computed: sensor type for UI method filtering
    const dt_image_t *img = &session->dev.image_storage;
    const gboolean is_xtrans = img->buf_dsc.filters == 9u;
    const gboolean is_bayer4 = img->flags & DT_IMAGE_4BAYER;
    const gboolean is_mono = dt_image_is_monochrome(img);
    const char *sensor_type = is_mono ? "mono" : is_xtrans ? "xtrans" : is_bayer4 ? "bayer4" : "bayer";
    json_builder_set_member_name(b, "sensor_type");
    json_builder_add_string_value(b, sensor_type);
  }
  else if(target->so->have_introspection && _introspection_serialize_params(b, target))
  {
    // Generic introspection-based serialization succeeded
  }
  else
  {
    // No introspection available — return raw base64 blob
    gchar *b64 = g_base64_encode((const guchar *)target->params, target->params_size);
    json_builder_set_member_name(b, "_raw_base64");
    json_builder_add_string_value(b, b64);
    json_builder_set_member_name(b, "_raw_size");
    json_builder_add_int_value(b, target->params_size);
    g_free(b64);
  }

  json_builder_end_object(b);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

// --- Async pipeline worker (event-driven preview) ---

typedef struct _pipeline_job_t
{
  dt_server_t *server;
  dt_server_session_t *session;
} _pipeline_job_t;

static void *_preview_pipeline_worker(void *arg)
{
  _pipeline_job_t *job = arg;
  dt_server_session_t *session = job->session;
  dt_server_t *server = job->server;
  g_free(job);

  while(TRUE)
  {
    // Snapshot the current seq before processing
    pthread_mutex_lock(&session->pipeline_mutex);
    const uint64_t start_seq = session->pipeline_seq;
    pthread_mutex_unlock(&session->pipeline_mutex);

    // Clear any pending shutdown flag before (re)processing
    dt_dev_pixelpipe_t *pipe_check = session->dev.full.pipe;
    if(pipe_check)
      dt_atomic_set_int(&pipe_check->shutdown, DT_DEV_PIXELPIPE_STOP_NO);

    // Process the pipeline
    const gint64 t_pipe_start = g_get_monotonic_time();
    dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
    dt_dev_process_image_job(&session->dev, &session->dev.full, pipe, -1, DT_DEVICE_CPU);
    const gint64 t_pipe_end = g_get_monotonic_time();
    fprintf(stderr, "[perf] pipeline: %.1f ms\n",
            (t_pipe_end - t_pipe_start) / 1000.0);

    // Check if new params arrived while we were processing
    pthread_mutex_lock(&session->pipeline_mutex);
    if(session->pipeline_seq != start_seq)
    {
      // New params arrived — loop and reprocess with latest values
      pthread_mutex_unlock(&session->pipeline_mutex);
      fprintf(stderr, "[perf] pipeline: stale, reprocessing\n");
      continue;
    }
    pthread_mutex_unlock(&session->pipeline_mutex);

    // Pipeline is current — write to SHM (IPC) or just signal (embedded) and send event
    dt_pthread_mutex_lock(&pipe->backbuf_mutex);

    if(!pipe->backbuf || pipe->backbuf_width <= 0 || pipe->backbuf_height <= 0)
    {
      dt_pthread_mutex_unlock(&pipe->backbuf_mutex);
      fprintf(stderr, "[server] async pipeline: no output\n");
      break;
    }

    const int rendered_width = pipe->backbuf_width;
    const int rendered_height = pipe->backbuf_height;

    session->frame_sequence++;

    if(!server->embedded && _ensure_shm_allocated(server, session))
    {
      // IPC mode: copy backbuf to SHM double-buffer
      const int back = 1 - session->front_buffer;
      dt_shm_buffer_t *shm = &session->shm_buffers[back];

      dt_shm_write_header(shm, rendered_width, rendered_height,
                           DT_SHM_FORMAT_BGRA8, session->frame_sequence);

      uint8_t *dst = dt_shm_pixel_data(shm);
      const size_t copy_size = (size_t)rendered_width * rendered_height * 4;
      memcpy(dst, pipe->backbuf, copy_size);

      __atomic_store_n(&shm->mapped->ready, 1, __ATOMIC_RELEASE);
      session->front_buffer = back;
    }
    // Embedded mode: backbuf stays in pipe, frame server reads it directly

    session->preview_width = rendered_width;
    session->preview_height = rendered_height;

    dt_pthread_mutex_unlock(&pipe->backbuf_mutex);

    // Queue preview_ready event
    {
      JsonBuilder *eb = json_builder_new();
      json_builder_begin_object(eb);
      json_builder_set_member_name(eb, "session_id");
      json_builder_add_string_value(eb, session->session_id);
      json_builder_set_member_name(eb, "front_buffer");
      json_builder_add_int_value(eb, session->front_buffer);
      json_builder_set_member_name(eb, "width");
      json_builder_add_int_value(eb, rendered_width);
      json_builder_set_member_name(eb, "height");
      json_builder_add_int_value(eb, rendered_height);
      json_builder_set_member_name(eb, "sequence");
      json_builder_add_int_value(eb, session->frame_sequence);
      json_builder_end_object(eb);

      JsonNode *event_data = json_builder_get_root(eb);
      dt_server_queue_event(server, "develop.preview_ready", event_data);
      json_node_unref(event_data);
      g_object_unref(eb);
    }

    {
      const gint64 t_event = g_get_monotonic_time();
      fprintf(stderr, "[perf] shm_write+event: %.1f ms  total_since_pipe_start: %.1f ms  %dx%d seq=%llu\n",
              (t_event - t_pipe_end) / 1000.0,
              (t_event - t_pipe_start) / 1000.0,
              rendered_width, rendered_height, (unsigned long long)session->frame_sequence);
    }
    break;
  }

  // Mark pipeline as no longer busy
  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_busy = FALSE;
  pthread_mutex_unlock(&session->pipeline_mutex);

  return NULL;
}

static void _maybe_start_pipeline(dt_server_t *server, dt_server_session_t *session)
{
  pthread_mutex_lock(&session->pipeline_mutex);
  if(session->pipeline_busy)
  {
    // Worker is already running — cancel the in-flight render so it finishes faster.
    // The worker will see the bumped pipeline_seq and reprocess with latest params.
    dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
    if(pipe)
      dt_atomic_set_int(&pipe->shutdown, DT_DEV_PIXELPIPE_STOP_NODES);
    pthread_mutex_unlock(&session->pipeline_mutex);
    return;
  }
  session->pipeline_busy = TRUE;
  pthread_mutex_unlock(&session->pipeline_mutex);

  _pipeline_job_t *job = g_new0(_pipeline_job_t, 1);
  job->server = server;
  job->session = session;

  pthread_t thread;
  if(dt_pthread_create(&thread, _preview_pipeline_worker, job) != 0)
  {
    fprintf(stderr, "[server] failed to create pipeline worker thread\n");
    g_free(job);
    pthread_mutex_lock(&session->pipeline_mutex);
    session->pipeline_busy = FALSE;
    pthread_mutex_unlock(&session->pipeline_mutex);
    return;
  }
  pthread_detach(thread);
}

char *dt_server_develop_set_params(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "params"))
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, op, or params");
  }

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  if(!session_id || !op)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id and op must be strings");
  JsonObject *new_params = json_object_get_object_member(req->params, "params");
  if(!new_params)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "params must be an object");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Find the module
  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  // Check for preview_only mode (skip history during drag)
  const gboolean preview_only = json_object_has_member(req->params, "preview_only")
    && json_object_get_boolean_member(req->params, "preview_only");

  // Handle enabled toggle (can be sent alone or with other params)
  gboolean enabled_only = FALSE;
  if(json_object_has_member(new_params, "enabled"))
  {
    target->enabled = json_object_get_boolean_member(new_params, "enabled");
    // Check if this is an enable-only request (no other params)
    if(json_object_get_size(new_params) == 1)
      enabled_only = TRUE;
  }

  // Apply module-specific parameter updates (skip if enable-only)
  if(enabled_only)
  {
    // No module params to update — just the enabled state
  }
  else if(!strcmp(op, "temperature"))
  {
    // Use introspection for basic field writes (red, green, blue, various, preset)
    _introspection_deserialize_params(target, new_params);

    _server_temperature_params_t *p = (_server_temperature_params_t *)target->params;

    // Post-processing: when preset changes, override coefficients from dev->chroma
    if(json_object_has_member(new_params, "preset"))
    {
      const int preset = p->preset;
      const dt_dev_chroma_t *chr = &session->dev.chroma;
      switch(preset)
      {
        case 0: // DT_IOP_TEMP_AS_SHOT
          p->red   = (float)(chr->as_shot[0] / chr->as_shot[1]);
          p->green = 1.0f;
          p->blue  = (float)(chr->as_shot[2] / chr->as_shot[1]);
          break;
        case 3: // DT_IOP_TEMP_D65 (camera reference)
          p->red   = (float)(chr->D65coeffs[0] / chr->D65coeffs[1]);
          p->green = 1.0f;
          p->blue  = (float)(chr->D65coeffs[2] / chr->D65coeffs[1]);
          break;
        case 4: // DT_IOP_TEMP_D65_LATE (as shot to reference)
          p->red   = (float)(chr->as_shot[0] / chr->as_shot[1]);
          p->green = 1.0f;
          p->blue  = (float)(chr->as_shot[2] / chr->as_shot[1]);
          break;
        default: // SPOT(1) or USER(2) — keep current coefficients
          break;
      }
    }

    // Handle temperature_k and/or tint: convert to RGB coefficients
    if(json_object_has_member(new_params, "temperature_k") || json_object_has_member(new_params, "tint"))
    {
      // Recover current temp/tint from coefficients using binary search
      double cur_temp_k = 5000.0, cur_tint = 1.0;
      {
        float d65_cm[9];
        memcpy(d65_cm, session->dev.image_storage.d65_color_matrix, sizeof(d65_cm));
        double _CAM_to_XYZ[3][4], _XYZ_to_CAM[4][3];
        if(dt_colorspaces_conversion_matrices_xyz(
             session->dev.image_storage.adobe_XYZ_to_CAM, d65_cm,
             _XYZ_to_CAM, _CAM_to_XYZ))
        {
          double CAM[4] = {
            p->red > 0.0f ? 1.0 / p->red : 0.0,
            p->green > 0.0f ? 1.0 / p->green : 0.0,
            p->blue > 0.0f ? 1.0 / p->blue : 0.0,
            p->various > 0.0f ? 1.0 / p->various : 0.0
          };
          double XYZ[3] = { 0, 0, 0 };
          for(int k = 0; k < 3; k++)
            for(int i = 0; i < 4; i++)
              XYZ[k] += _CAM_to_XYZ[k][i] * CAM[i];

          if(XYZ[0] > 0 && XYZ[1] > 0 && XYZ[2] > 0)
          {
            cmsCIEXYZ cmsXYZ = { XYZ[0], XYZ[1], XYZ[2] };
            float ft, fti;
            _server_XYZ_to_temperature(cmsXYZ, &ft, &fti);
            cur_temp_k = ft;
            cur_tint = fti;
          }
        }
      }

      double new_temp_k = json_object_has_member(new_params, "temperature_k")
        ? json_object_get_double_member(new_params, "temperature_k") : cur_temp_k;
      double new_tint = json_object_has_member(new_params, "tint")
        ? json_object_get_double_member(new_params, "tint") : cur_tint;

      double mul[4];
      if(_server_temp_tint_to_mul(&session->dev.image_storage, new_temp_k, new_tint, mul)
         && isfinite(mul[0]) && isfinite(mul[1]) && isfinite(mul[2])
         && mul[0] > 0.0 && mul[1] > 0.0 && mul[2] > 0.0)
      {
        fprintf(stderr, "[server] temp_tint_to_mul: temp=%.0f tint=%.3f -> R=%.4f G=%.4f B=%.4f\n",
                new_temp_k, new_tint, mul[0], mul[1], mul[2]);
        p->red     = (float)mul[0];
        p->green   = (float)mul[1];
        p->blue    = (float)mul[2];
        if(p->various > 0.0f && isfinite(mul[3]) && mul[3] > 0.0)
          p->various = (float)mul[3];
        p->preset  = 2; // DT_IOP_TEMP_USER — user modified
      }
      else
      {
        fprintf(stderr, "[server] temp_tint_to_mul FAILED: temp=%.0f tint=%.3f\n",
                new_temp_k, new_tint);
      }
    }
  }
  else if(target->so->have_introspection && _introspection_deserialize_params(target, new_params))
  {
    // Generic introspection-based deserialization succeeded
  }
  else
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Module has no introspection data for parameter updates");
  }

  if(preview_only)
  {
    // Preview-only: commit params directly to the pipe piece, bypassing history.
    // DT_DEV_PIPE_TOP_CHANGED triggers synch_top() which reads dev->history —
    // since we skipped history write, that would revert our changes.
    // Instead, commit to the pipe piece directly and invalidate its cache.
    dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
    for(GList *nodes = pipe->nodes; nodes; nodes = g_list_next(nodes))
    {
      dt_dev_pixelpipe_iop_t *piece = nodes->data;
      if(piece->module == target)
      {
        piece->enabled = target->enabled;
        dt_iop_commit_params(target, target->params,
                             target->blend_params ? target->blend_params
                               : target->default_blendop_params,
                             pipe, piece);
        // Invalidate cache from this module onwards
        dt_dev_pixelpipe_cache_invalidate_later(pipe, target->iop_order);
        break;
      }
    }
  }
  else
  {
    // Record the change to history.
    // Use _ext variant which bypasses the GUI check in dt_dev_add_history_item()
    // (the wrapper bails early when darktable.gui is NULL, i.e. headless/server mode).
    // no_image=TRUE avoids GUI widget operations that crash without a GUI.
    dt_dev_add_history_item_ext(&session->dev, target, target->enabled, TRUE);

    // Mark pipeline dirty — no_image=TRUE above skips pipe->changed, so set manually
    session->dev.full.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
    if(session->dev.preview_pipe)
      session->dev.preview_pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
    if(session->dev.preview2.pipe)
      session->dev.preview2.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  }

  dt_dev_invalidate_all(&session->dev);
  session->dirty = TRUE;

  // Bump pipeline sequence and trigger async processing
  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_seq++;
  pthread_mutex_unlock(&session->pipeline_mutex);
  _maybe_start_pipeline(server, session);

  fprintf(stderr, "[server] develop.set_params: session=%s op=%s enabled=%d history_end=%d\n",
          session_id, op, target->enabled, session->dev.history_end);

  // Build response
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_set_member_name(b, "dirty");
  json_builder_add_boolean_value(b, session->dirty);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_request_preview(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Async preview: bump pipeline_seq and trigger worker thread.
  // The worker writes to SHM and pushes a develop.preview_ready event on completion.
  // The server main loop returns immediately so it can keep processing other requests.
  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_seq++;
  pthread_mutex_unlock(&session->pipeline_mutex);
  _maybe_start_pipeline(server, session);

  fprintf(stderr, "[server] develop.request_preview: session=%s queued (seq=%llu)\n",
          session_id, (unsigned long long)session->pipeline_seq);

  // Respond immediately -- client will receive develop.preview_ready event when done
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "queued");

  json_builder_set_member_name(b, "session_id");
  json_builder_add_string_value(b, session->session_id);

  json_builder_set_member_name(b, "pipeline_seq");
  json_builder_add_int_value(b, session->pipeline_seq);

  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_cancel_pipeline(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  /* Signal the pipeline to stop at the next IOP boundary */
  dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
  if(pipe)
    dt_atomic_set_int(&pipe->shutdown, DT_DEV_PIXELPIPE_STOP_NODES);

  fprintf(stderr, "[server] develop.cancel_pipeline: session=%s\n", session_id);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "cancelled");
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_sample_pixels(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "x")
     || !json_object_has_member(req->params, "y")
     || !json_object_has_member(req->params, "w")
     || !json_object_has_member(req->params, "h"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, x, y, w, h");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");

  const double nx = json_object_get_double_member(req->params, "x");
  const double ny = json_object_get_double_member(req->params, "y");
  const double nw = json_object_get_double_member(req->params, "w");
  const double nh = json_object_get_double_member(req->params, "h");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;

  dt_pthread_mutex_lock(&pipe->backbuf_mutex);

  if(!pipe->backbuf || pipe->backbuf_width <= 0 || pipe->backbuf_height <= 0)
  {
    dt_pthread_mutex_unlock(&pipe->backbuf_mutex);
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "No backbuf available");
  }

  const int bw = pipe->backbuf_width;
  const int bh = pipe->backbuf_height;
  const uint8_t *buf = pipe->backbuf;

  // Convert normalized coords to pixel coords, clamped
  int px = (int)(nx * bw);
  int py = (int)(ny * bh);
  int pw = (int)(nw * bw);
  int ph = (int)(nh * bh);
  if(px < 0) px = 0;
  if(py < 0) py = 0;
  if(px + pw > bw) pw = bw - px;
  if(py + ph > bh) ph = bh - py;

  double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
  int count = 0;

  for(int y = py; y < py + ph; y++)
  {
    const uint8_t *row = buf + (size_t)y * bw * 4 + (size_t)px * 4;
    for(int x = 0; x < pw; x++)
    {
      // BGRA8 format
      sum_b += row[0];
      sum_g += row[1];
      sum_r += row[2];
      row += 4;
      count++;
    }
  }

  dt_pthread_mutex_unlock(&pipe->backbuf_mutex);

  if(count == 0)
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "Empty sample area");

  const double mean_r = sum_r / count / 255.0;
  const double mean_g = sum_g / count / 255.0;
  const double mean_b = sum_b / count / 255.0;

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "mean_r");
  json_builder_add_double_value(b, mean_r);
  json_builder_set_member_name(b, "mean_g");
  json_builder_add_double_value(b, mean_g);
  json_builder_set_member_name(b, "mean_b");
  json_builder_add_double_value(b, mean_b);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_commit_params(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op"))
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or op");
  }

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  if(!session_id || !op)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id and op must be strings");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Find the module
  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  // Write current params to history (the params were already applied by preview_only set_params calls)
  dt_dev_add_history_item_ext(&session->dev, target, target->enabled, TRUE);
  session->dirty = TRUE;

  fprintf(stderr, "[server] develop.commit_params: session=%s op=%s history_end=%d\n",
          session_id, op, session->dev.history_end);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_reset_params(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op"))
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or op");
  }

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  if(!session_id || !op)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id and op must be strings");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Find the module
  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  // Reset params to defaults
  memcpy(target->params, target->default_params, target->params_size);

  // Record the change to history
  dt_dev_add_history_item_ext(&session->dev, target, target->enabled, TRUE);

  // Mark pipeline dirty
  session->dev.full.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  if(session->dev.preview_pipe)
    session->dev.preview_pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  if(session->dev.preview2.pipe)
    session->dev.preview2.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;

  dt_dev_invalidate_all(&session->dev);
  session->dirty = TRUE;

  // Bump pipeline sequence and trigger async processing
  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_seq++;
  pthread_mutex_unlock(&session->pipeline_mutex);
  _maybe_start_pipeline(server, session);

  fprintf(stderr, "[server] develop.reset_params: session=%s op=%s history_end=%d\n",
          session_id, op, session->dev.history_end);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_set_member_name(b, "dirty");
  json_builder_add_boolean_value(b, session->dirty);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_select_history(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "history_end"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id or history_end parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");

  const int history_end = (int)json_object_get_int_member(req->params, "history_end");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_develop_t *dev = &session->dev;

  // Pop history to the requested point
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_pop_history_items_ext(dev, history_end);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  // Mark pipe dirty for re-rendering
  if(dev->full.pipe)
  {
    dev->full.pipe->changed |= DT_DEV_PIPE_REMOVE;
    dev->full.pipe->status = DT_DEV_PIXELPIPE_DIRTY;
  }

  dt_dev_invalidate_all(dev);
  session->dirty = TRUE;

  // Trigger async pipeline processing
  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_seq++;
  pthread_mutex_unlock(&session->pipeline_mutex);
  _maybe_start_pipeline(server, session);

  fprintf(stderr, "[server] develop.select_history: session=%s history_end=%d\n",
          session_id, dev->history_end);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_set_member_name(b, "history_end");
  json_builder_add_int_value(b, dev->history_end);
  json_builder_end_object(b);

  JsonNode *res = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, res);
  json_node_unref(res);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_compress_history(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  const dt_imgid_t imgid = session->imgid;
  dt_develop_t *dev = &session->dev;

  // Write current in-memory history to DB before compressing
  dt_dev_write_history(dev);

  // Compress history in the database
  dt_history_compress_on_image(imgid);

  dt_lock_image(imgid);

  // Reset all modules to defaults
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_pop_history_items_ext(dev, 0);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  // Remove in-memory history items
  GList *history = dev->history;
  while(history)
  {
    GList *next = g_list_next(history);
    dt_dev_history_item_t *hist = history->data;
    dt_dev_free_history_item(hist);
    dev->history = g_list_delete_link(dev->history, history);
    history = next;
  }

  // Re-read compressed history from DB
  dt_dev_read_history(dev);

  // Apply the compressed history
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_pop_history_items_ext(dev, dev->history_end);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  dt_ioppr_resync_iop_list(dev);

  // Mark pipe dirty for re-rendering
  if(dev->full.pipe)
  {
    dev->full.pipe->changed |= DT_DEV_PIPE_REMOVE;
    dev->full.pipe->status = DT_DEV_PIXELPIPE_DIRTY;
  }

  dt_unlock_image(imgid);

  // Trigger async pipeline processing
  dt_dev_invalidate_all(dev);
  session->dirty = TRUE;
  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_seq++;
  pthread_mutex_unlock(&session->pipeline_mutex);
  _maybe_start_pipeline(server, session);

  fprintf(stderr, "[server] develop.compress_history: session=%s history_end=%d\n",
          session_id, dev->history_end);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_end_object(b);

  JsonNode *res = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, res);
  json_node_unref(res);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_truncate_history(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "history_end"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id or history_end parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");

  const int history_end = (int)json_object_get_int_member(req->params, "history_end");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  const dt_imgid_t imgid = session->imgid;
  dt_develop_t *dev = &session->dev;

  // Write current in-memory history to DB before truncating
  dt_dev_write_history(dev);

  // Truncate history in the database (deletes entries with num >= history_end)
  dt_history_truncate_on_image(imgid, history_end);

  dt_lock_image(imgid);

  // Reset all modules to defaults
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_pop_history_items_ext(dev, 0);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  // Remove in-memory history items
  GList *history = dev->history;
  while(history)
  {
    GList *next = g_list_next(history);
    dt_dev_history_item_t *hist = history->data;
    dt_dev_free_history_item(hist);
    dev->history = g_list_delete_link(dev->history, history);
    history = next;
  }

  // Re-read truncated history from DB
  dt_dev_read_history(dev);

  // Apply the truncated history
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_pop_history_items_ext(dev, dev->history_end);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  dt_ioppr_resync_iop_list(dev);

  // Mark pipe dirty for re-rendering
  if(dev->full.pipe)
  {
    dev->full.pipe->changed |= DT_DEV_PIPE_REMOVE;
    dev->full.pipe->status = DT_DEV_PIXELPIPE_DIRTY;
  }

  dt_unlock_image(imgid);

  // Trigger async pipeline processing
  dt_dev_invalidate_all(dev);
  session->dirty = TRUE;
  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_seq++;
  pthread_mutex_unlock(&session->pipeline_mutex);
  _maybe_start_pipeline(server, session);

  fprintf(stderr, "[server] develop.truncate_history: session=%s history_end=%d\n",
          session_id, dev->history_end);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_set_member_name(b, "history_end");
  json_builder_add_int_value(b, dev->history_end);
  json_builder_end_object(b);

  JsonNode *res = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, res);
  json_node_unref(res);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_delete_history(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  const dt_imgid_t imgid = session->imgid;

  // Delete history (no undo in server mode).
  // Pass init_history=FALSE because the TRUE path calls
  // dt_dev_reload_history_items(darktable.develop) which uses GUI code.
  dt_history_delete_on_image_ext(imgid, FALSE, FALSE);

  // Clear auto-presets-applied flag so dt_dev_read_history re-applies
  // default modules (sigmoid, etc.). This is what _remove_preset_flag()
  // does in the init_history=TRUE path.
  {
    dt_image_t *image = dt_image_cache_get(imgid, 'w');
    if(image)
    {
      image->flags &= ~DT_IMAGE_AUTO_PRESETS_APPLIED;
      dt_image_cache_write_release_info(image, DT_IMAGE_CACHE_SAFE,
                                        "server_delete_history");
    }
  }

  // Server-safe reload: dt_dev_reload_history_items() calls GUI code,
  // so we do the non-GUI parts inline here.
  dt_develop_t *dev = &session->dev;
  dev->focus_hash = FALSE;

  dt_lock_image(imgid);

  // Reset all modules to defaults
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_pop_history_items_ext(dev, 0);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  // Remove unused history items from the in-memory list
  GList *history = g_list_nth(dev->history, dev->history_end);
  while(history)
  {
    GList *next = g_list_next(history);
    dt_dev_history_item_t *hist = history->data;
    hist->module->multi_name_hand_edited = FALSE;
    g_strlcpy(hist->module->multi_name, "", sizeof(hist->module->multi_name));
    dt_dev_free_history_item(hist);
    dev->history = g_list_delete_link(dev->history, history);
    history = next;
  }

  // Re-read history from the now-clean database (will re-apply auto-presets)
  dt_dev_read_history(dev);
  dt_ioppr_set_default_iop_order(dev, imgid);

  // Apply the (now clean) history
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_pop_history_items_ext(dev, dev->history_end);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  dt_ioppr_resync_iop_list(dev);

  // Mark pipe dirty for re-rendering
  if(dev->full.pipe)
  {
    dev->full.pipe->changed |= DT_DEV_PIPE_REMOVE;
    dev->full.pipe->status = DT_DEV_PIXELPIPE_DIRTY;
  }

  dt_unlock_image(imgid);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_end_object(b);

  JsonNode *res = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, res);
  json_node_unref(res);
  g_object_unref(b);
  return resp;
}

// ──────────────────────────────────────────────────────────────────────────────
// Preset handlers
// ──────────────────────────────────────────────────────────────────────────────

char *dt_server_develop_list_presets(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or op");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  if(!session_id || !op)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id and op must be strings");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  // Find the module to compare current params for active detection
  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  // Query all presets for this operation
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "SELECT name, description, op_params, blendop_params, enabled, writeprotect"
    " FROM data.presets"
    " WHERE operation = ?1 AND op_version = ?2"
    " ORDER BY writeprotect DESC, LOWER(name), rowid",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, target->op, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, target->version());

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "presets");
  json_builder_begin_array(b);

  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const char *name = (const char *)sqlite3_column_text(stmt, 0);
    const char *description = (const char *)sqlite3_column_text(stmt, 1);
    const void *op_params = sqlite3_column_blob(stmt, 2);
    const int op_length = sqlite3_column_bytes(stmt, 2);
    const void *blendop_params = sqlite3_column_blob(stmt, 3);
    const int bl_length = sqlite3_column_bytes(stmt, 3);
    const int enabled = sqlite3_column_int(stmt, 4);
    const int writeprotect = sqlite3_column_int(stmt, 5);

    // Check if this preset matches current module state (active detection)
    gboolean is_active = FALSE;
    if(((op_length == 0
         && !memcmp(target->default_params, target->params, target->params_size))
        || ((op_length > 0
             && op_length == (int)target->params_size
             && !memcmp(target->params, op_params, op_length))))
       && blendop_params
       && bl_length == (int)sizeof(dt_develop_blend_params_t)
       && !memcmp(target->blend_params, blendop_params, bl_length)
       && target->enabled == enabled)
    {
      is_active = TRUE;
    }

    json_builder_begin_object(b);
    json_builder_set_member_name(b, "name");
    json_builder_add_string_value(b, name ? name : "");
    json_builder_set_member_name(b, "description");
    json_builder_add_string_value(b, description ? description : "");
    json_builder_set_member_name(b, "writeprotect");
    json_builder_add_boolean_value(b, writeprotect);
    json_builder_set_member_name(b, "active");
    json_builder_add_boolean_value(b, is_active);
    json_builder_end_object(b);
  }
  sqlite3_finalize(stmt);

  json_builder_end_array(b);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_apply_preset(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "name"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, op, or name");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  const char *name = json_object_get_string_member(req->params, "name");
  if(!session_id || !op || !name)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "params must be strings");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  // Fetch preset from database (mirrors dt_gui_presets_apply_preset logic)
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "SELECT op_params, enabled, blendop_params, blendop_version"
    " FROM data.presets"
    " WHERE operation = ?1 AND op_version = ?2 AND name = ?3",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, target->op, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, target->version());
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, name, -1, SQLITE_TRANSIENT);

  if(sqlite3_step(stmt) != SQLITE_ROW)
  {
    sqlite3_finalize(stmt);
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Preset not found");
  }

  const void *op_params = sqlite3_column_blob(stmt, 0);
  const int op_length = sqlite3_column_bytes(stmt, 0);
  const int enabled = sqlite3_column_int(stmt, 1);
  const void *blendop_params = sqlite3_column_blob(stmt, 2);
  const int bl_length = sqlite3_column_bytes(stmt, 2);
  const int blendop_version = sqlite3_column_int(stmt, 3);

  // Apply op_params
  if(op_params && (op_length == (int)target->params_size))
    memcpy(target->params, op_params, op_length);
  else
    memcpy(target->params, target->default_params, target->params_size);

  target->enabled = enabled;

  // Apply blend params
  if(blendop_params
     && (blendop_version == dt_develop_blend_version())
     && (bl_length == (int)sizeof(dt_develop_blend_params_t)))
  {
    dt_iop_commit_blend_params(target, blendop_params);
  }
  else if(blendop_params
          && dt_develop_blend_legacy_params(target, blendop_params,
                                            blendop_version, target->blend_params,
                                            dt_develop_blend_version(), bl_length) == FALSE)
  {
    // legacy conversion succeeded — blend_params already updated
  }
  else
  {
    dt_iop_commit_blend_params(target, target->default_blendop_params);
  }

  sqlite3_finalize(stmt);

  // Record to history
  dt_dev_add_history_item_ext(&session->dev, target, target->enabled, TRUE);

  // Mark pipeline dirty
  session->dev.full.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  if(session->dev.preview_pipe)
    session->dev.preview_pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  if(session->dev.preview2.pipe)
    session->dev.preview2.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;

  dt_dev_invalidate_all(&session->dev);
  session->dirty = TRUE;

  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_seq++;
  pthread_mutex_unlock(&session->pipeline_mutex);
  _maybe_start_pipeline(server, session);

  fprintf(stderr, "[server] develop.apply_preset: session=%s op=%s preset='%s'\n",
          session_id, op, name);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_set_member_name(b, "dirty");
  json_builder_add_boolean_value(b, session->dirty);
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_store_preset(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "name"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, op, or name");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  const char *name = json_object_get_string_member(req->params, "name");
  const char *description = json_object_has_member(req->params, "description")
    ? json_object_get_string_member(req->params, "description") : "";
  if(!session_id || !op || !name)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "params must be strings");

  // Optional filter fields
  const int autoapply = json_object_has_member(req->params, "autoapply")
    ? (int)json_object_get_int_member(req->params, "autoapply") : 0;
  const int filter = json_object_has_member(req->params, "filter")
    ? (int)json_object_get_int_member(req->params, "filter") : 0;
  const char *p_model = json_object_has_member(req->params, "model")
    ? json_object_get_string_member(req->params, "model") : "%";
  const char *p_maker = json_object_has_member(req->params, "maker")
    ? json_object_get_string_member(req->params, "maker") : "%";
  const char *p_lens = json_object_has_member(req->params, "lens")
    ? json_object_get_string_member(req->params, "lens") : "%";
  const double iso_min = json_object_has_member(req->params, "iso_min")
    ? json_object_get_double_member(req->params, "iso_min") : 0;
  const double iso_max = json_object_has_member(req->params, "iso_max")
    ? json_object_get_double_member(req->params, "iso_max") : 51200;
  const double exposure_min = json_object_has_member(req->params, "exposure_min")
    ? json_object_get_double_member(req->params, "exposure_min") : 0;
  const double exposure_max = json_object_has_member(req->params, "exposure_max")
    ? json_object_get_double_member(req->params, "exposure_max") : 10000;
  const double aperture_min = json_object_has_member(req->params, "aperture_min")
    ? json_object_get_double_member(req->params, "aperture_min") : 0;
  const double aperture_max = json_object_has_member(req->params, "aperture_max")
    ? json_object_get_double_member(req->params, "aperture_max") : 128;
  const double focal_length_min = json_object_has_member(req->params, "focal_length_min")
    ? json_object_get_double_member(req->params, "focal_length_min") : 0;
  const double focal_length_max = json_object_has_member(req->params, "focal_length_max")
    ? json_object_get_double_member(req->params, "focal_length_max") : 1000;
  const int format = json_object_has_member(req->params, "format")
    ? (int)json_object_get_int_member(req->params, "format") : 0x1f;

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  // Check if a writeprotected preset with this name already exists
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "SELECT writeprotect FROM data.presets"
    " WHERE operation = ?1 AND op_version = ?2 AND name = ?3",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, target->op, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, target->version());
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, name, -1, SQLITE_TRANSIENT);

  if(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const int wp = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    if(wp)
      return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                   "Cannot overwrite write-protected preset");

    // Delete existing user preset so we can replace it
    DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "DELETE FROM data.presets WHERE operation = ?1 AND op_version = ?2 AND name = ?3",
      -1, &stmt, NULL);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, target->op, -1, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, target->version());
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, name, -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
  }
  else
  {
    sqlite3_finalize(stmt);
  }

  // Insert new preset with current module params and filter settings
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "INSERT INTO data.presets"
    " (name, description, operation, op_version, op_params, enabled,"
    "  blendop_params, blendop_version, writeprotect,"
    "  autoapply, filter, model, maker, lens,"
    "  iso_min, iso_max, exposure_min, exposure_max,"
    "  aperture_min, aperture_max, focal_length_min, focal_length_max, format)"
    " VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0,"
    "         ?9, ?10, ?11, ?12, ?13,"
    "         ?14, ?15, ?16, ?17,"
    "         ?18, ?19, ?20, ?21, ?22)",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, description ? description : "", -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, target->op, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 4, target->version());
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 5, target->params, target->params_size, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 6, target->enabled);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 7, target->blend_params, sizeof(dt_develop_blend_params_t), SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 8, dt_develop_blend_version());
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 9, autoapply);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 10, filter);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 11, p_model, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 12, p_maker, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 13, p_lens, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 14, iso_min);
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 15, iso_max);
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 16, exposure_min);
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 17, exposure_max);
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 18, aperture_min);
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 19, aperture_max);
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 20, focal_length_min);
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 21, focal_length_max);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 22, format);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  fprintf(stderr, "[server] develop.store_preset: session=%s op=%s name='%s'\n",
          session_id, op, name);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

// ---- Multi-instance helpers ----

/** Find a module by op + multi_priority (instance) */
static dt_iop_module_t *_find_module(dt_server_session_t *session, const char *op, int instance)
{
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op) && mod->multi_priority == instance)
      return mod;
  }
  return NULL;
}

/** Count instances sharing the same base module (by instance pointer) */
static int _count_instances(dt_develop_t *dev, dt_iop_module_t *module)
{
  int count = 0;
  for(GList *modules = dev->iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(mod->instance == module->instance) count++;
  }
  return count;
}

/** Get previous module in iop list order (for move down) */
static dt_iop_module_t *_get_prev_module(dt_develop_t *dev, dt_iop_module_t *module)
{
  dt_iop_module_t *prev = NULL;
  for(GList *modules = dev->iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(mod == module) break;
    prev = mod;
  }
  return prev;
}

/** Get next module in iop list order (for move up) */
static dt_iop_module_t *_get_next_module(dt_develop_t *dev, dt_iop_module_t *module)
{
  gboolean found = FALSE;
  for(GList *modules = dev->iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(found) return mod;
    if(mod == module) found = TRUE;
  }
  return NULL;
}

/** Build a JSON ok response with updated modules list */
static char *_make_modules_response(dt_server_t *server, dt_server_session_t *session,
                                    const dt_server_request_t *req)
{
  // Trigger pipeline rebuild
  session->dev.full.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  if(session->dev.preview_pipe)
    session->dev.preview_pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  if(session->dev.preview2.pipe)
    session->dev.preview2.pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;

  dt_dev_invalidate_all(&session->dev);
  session->dirty = TRUE;

  // Persist history to database
  dt_dev_write_history(&session->dev);

  pthread_mutex_lock(&session->pipeline_mutex);
  session->pipeline_seq++;
  pthread_mutex_unlock(&session->pipeline_mutex);
  _maybe_start_pipeline(server, session);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

// ---- develop.new_instance / develop.duplicate_instance ----

char *dt_server_develop_new_instance(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id or op");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  const int instance = json_object_has_member(req->params, "instance")
    ? (int)json_object_get_int_member(req->params, "instance") : 0;
  const gboolean copy_params = json_object_has_member(req->params, "copy_params")
    && json_object_get_boolean_member(req->params, "copy_params");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_iop_module_t *base = _find_module(session, op, instance);
  if(!base)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  if(base->flags() & IOP_FLAGS_ONE_INSTANCE)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Module does not support multiple instances");

  // Ensure base has a history entry before duplicating
  dt_dev_add_history_item_ext(&session->dev, base, base->enabled, TRUE);

  // Create the new module instance
  dt_iop_module_t *module = dt_dev_module_duplicate(&session->dev, base);
  if(!module)
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "Failed to create module instance");

  if(copy_params)
  {
    // Duplicate: copy params and enabled state from base
    memcpy(module->params, base->params, module->params_size);
    module->enabled = base->enabled;
    if(module->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
    {
      dt_iop_commit_blend_params(module, base->blend_params);
      if(dt_is_valid_maskid(base->blend_params->mask_id))
      {
        module->blend_params->mask_id = NO_MASKID;
      }
    }
  }
  else
  {
    // New instance: enable by default
    module->enabled = TRUE;
  }

  // Save the new instance creation to history
  dt_dev_add_history_item_ext(&session->dev, module, module->enabled, TRUE);

  fprintf(stderr, "[server] develop.new_instance: op=%s instance=%d copy=%d new_instance=%d\n",
          op, instance, copy_params, module->multi_priority);

  return _make_modules_response(server, session, req);
}

// ---- develop.delete_instance ----

char *dt_server_develop_delete_instance(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "instance"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id, op, or instance");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  const int instance = (int)json_object_get_int_member(req->params, "instance");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_iop_module_t *module = _find_module(session, op, instance);
  if(!module)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  const int nb_instances = _count_instances(&session->dev, module);
  if(nb_instances <= 1)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Cannot delete the only instance");

  // Remember if this was priority 0
  const gboolean is_zero = (module->multi_priority == 0);

  // Remove from history and iop list
  dt_dev_module_remove(&session->dev, module);

  // If deleted module was priority 0, reassign priority 0 to another instance
  if(is_zero)
  {
    dt_iop_module_t *first = NULL;
    for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
    {
      dt_iop_module_t *mod = modules->data;
      if(mod->instance == module->instance)
      {
        first = mod;
        break;
      }
    }
    if(first)
    {
      dt_iop_update_multi_priority(first, 0);
      // Update in history too
      for(GList *history = session->dev.history; history; history = g_list_next(history))
      {
        dt_dev_history_item_t *hist = history->data;
        if(hist->module == first) hist->multi_priority = 0;
      }
    }
  }

  // Don't free the module — pipeline may still reference it
  session->dev.alliop = g_list_append(session->dev.alliop, module);

  fprintf(stderr, "[server] develop.delete_instance: op=%s instance=%d\n", op, instance);

  return _make_modules_response(server, session, req);
}

// ---- develop.move_instance ----

char *dt_server_develop_move_instance(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "instance")
     || !json_object_has_member(req->params, "direction"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, op, instance, or direction");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  const int instance = (int)json_object_get_int_member(req->params, "instance");
  const char *direction = json_object_get_string_member(req->params, "direction");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_iop_module_t *module = _find_module(session, op, instance);
  if(!module)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  int moved = 0;
  if(!strcmp(direction, "up"))
  {
    dt_iop_module_t *next = _get_next_module(&session->dev, module);
    if(next)
    {
      moved = dt_ioppr_move_iop_after(&session->dev, module, next);
    }
  }
  else if(!strcmp(direction, "down"))
  {
    dt_iop_module_t *prev = _get_prev_module(&session->dev, module);
    if(prev)
    {
      moved = dt_ioppr_move_iop_before(&session->dev, module, prev);
    }
  }
  else
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "direction must be 'up' or 'down'");
  }

  if(!moved)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Cannot move module in that direction");

  dt_dev_add_history_item_ext(&session->dev, module, module->enabled, TRUE);

  fprintf(stderr, "[server] develop.move_instance: op=%s instance=%d direction=%s\n",
          op, instance, direction);

  return _make_modules_response(server, session, req);
}

// ---- develop.rename_instance ----

char *dt_server_develop_rename_instance(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "instance")
     || !json_object_has_member(req->params, "name"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, op, instance, or name");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  const int instance = (int)json_object_get_int_member(req->params, "instance");
  const char *name = json_object_get_string_member(req->params, "name");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_iop_module_t *module = _find_module(session, op, instance);
  if(!module)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  g_strlcpy(module->multi_name, name, sizeof(module->multi_name));
  module->multi_name_hand_edited = TRUE;

  dt_dev_add_history_item_ext(&session->dev, module, module->enabled, TRUE);

  fprintf(stderr, "[server] develop.rename_instance: op=%s instance=%d name='%s'\n",
          op, instance, name);

  return _make_modules_response(server, session, req);
}

// ---- develop.set_blend_param ----

char *dt_server_develop_set_blend_param(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "param")
     || !json_object_has_member(req->params, "value"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, op, param, or value");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  const int instance = json_object_has_member(req->params, "instance")
    ? (int)json_object_get_int_member(req->params, "instance") : 0;
  const char *param = json_object_get_string_member(req->params, "param");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_iop_module_t *module = _find_module(session, op, instance);
  if(!module)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  if(!(module->flags() & IOP_FLAGS_SUPPORTS_BLENDING) || !module->blend_params)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Module does not support blending");

  dt_develop_blend_params_t *bp = module->blend_params;

  if(!strcmp(param, "mask_mode"))
    bp->mask_mode = (uint32_t)json_object_get_int_member(req->params, "value");
  else if(!strcmp(param, "blend_mode"))
    bp->blend_mode = (uint32_t)json_object_get_int_member(req->params, "value");
  else if(!strcmp(param, "opacity"))
    bp->opacity = (float)json_object_get_double_member(req->params, "value");
  else if(!strcmp(param, "mask_combine"))
    bp->mask_combine = (uint32_t)json_object_get_int_member(req->params, "value");
  else if(!strcmp(param, "blend_parameter"))
    bp->blend_parameter = (float)json_object_get_double_member(req->params, "value");
  else if(!strcmp(param, "details"))
    bp->details = (float)json_object_get_double_member(req->params, "value");
  else if(!strcmp(param, "feathering_guide"))
    bp->feathering_guide = (uint32_t)json_object_get_int_member(req->params, "value");
  else if(!strcmp(param, "feathering_radius"))
    bp->feathering_radius = (float)json_object_get_double_member(req->params, "value");
  else if(!strcmp(param, "blur_radius"))
    bp->blur_radius = (float)json_object_get_double_member(req->params, "value");
  else if(!strcmp(param, "contrast"))
    bp->contrast = (float)json_object_get_double_member(req->params, "value");
  else if(!strcmp(param, "brightness"))
    bp->brightness = (float)json_object_get_double_member(req->params, "value");
  else
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Unknown blend param");

  dt_iop_commit_blend_params(module, bp);
  dt_dev_add_history_item_ext(&session->dev, module, module->enabled, TRUE);

  fprintf(stderr, "[server] develop.set_blend_param: op=%s instance=%d param=%s\n",
          op, instance, param);

  return _make_modules_response(server, session, req);
}

char *dt_server_develop_delete_preset(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "op")
     || !json_object_has_member(req->params, "name"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing op or name");

  const char *op = json_object_get_string_member(req->params, "op");
  const char *name = json_object_get_string_member(req->params, "name");
  if(!op || !name)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "op and name must be strings");

  // Check writeprotect
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "SELECT writeprotect FROM data.presets"
    " WHERE name = ?1 AND operation = ?2",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, op, -1, SQLITE_TRANSIENT);

  if(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const int wp = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    if(wp)
      return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                   "Cannot delete write-protected preset");
  }
  else
  {
    sqlite3_finalize(stmt);
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Preset not found");
  }

  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "DELETE FROM data.presets WHERE name = ?1 AND operation = ?2",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, op, -1, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  fprintf(stderr, "[server] develop.delete_preset: op=%s name='%s'\n", op, name);

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

char *dt_server_develop_get_introspection(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "op"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or op parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  const char *op = json_object_get_string_member(req->params, "op");
  if(!session_id || !op)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id and op must be strings");

  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_iop_module_t *target = NULL;
  for(GList *modules = session->dev.iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = modules->data;
    if(dt_iop_module_is(mod->so, op))
    {
      target = mod;
      break;
    }
  }

  if(!target)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");

  if(!target->so->have_introspection)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Module has no introspection");

  dt_introspection_t *intro = target->so->get_introspection();
  if(!intro || !intro->field || intro->field->header.type != DT_INTROSPECTION_TYPE_STRUCT)
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "Invalid introspection data");

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "op");
  json_builder_add_string_value(b, op);

  json_builder_set_member_name(b, "params_version");
  json_builder_add_int_value(b, intro->params_version);

  json_builder_set_member_name(b, "fields");
  json_builder_begin_array(b);
  const dt_introspection_type_struct_t *root = &intro->field->Struct;
  for(size_t i = 0; i < root->entries; i++)
    _introspection_serialize_schema_field(b, root->fields[i]);
  json_builder_end_array(b);

  json_builder_end_object(b);

  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

static const char *_mask_type_name(dt_masks_type_t type)
{
  const int base = type & ~(DT_MASKS_CLONE | DT_MASKS_NON_CLONE);
  switch(base)
  {
    case DT_MASKS_CIRCLE:   return "circle";
    case DT_MASKS_ELLIPSE:  return "ellipse";
    case DT_MASKS_PATH:     return "path";
    case DT_MASKS_GRADIENT: return "gradient";
    case DT_MASKS_BRUSH:    return "brush";
    case DT_MASKS_GROUP:    return "group";
    default:                return "unknown";
  }
}

char *dt_server_develop_get_masks(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "forms");
  json_builder_begin_array(b);

  for(GList *l = session->dev.forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *form = l->data;
    if(!form) continue;

    json_builder_begin_object(b);

    json_builder_set_member_name(b, "formid");
    json_builder_add_int_value(b, form->formid);

    json_builder_set_member_name(b, "name");
    json_builder_add_string_value(b, form->name);

    json_builder_set_member_name(b, "type");
    json_builder_add_int_value(b, form->type);

    json_builder_set_member_name(b, "type_name");
    json_builder_add_string_value(b, _mask_type_name(form->type));

    json_builder_set_member_name(b, "is_clone");
    json_builder_add_boolean_value(b, (form->type & DT_MASKS_CLONE) != 0);

    // Source position for clone masks
    if(form->type & DT_MASKS_CLONE)
    {
      json_builder_set_member_name(b, "source");
      json_builder_begin_array(b);
      json_builder_add_double_value(b, form->source[0]);
      json_builder_add_double_value(b, form->source[1]);
      json_builder_end_array(b);
    }

    // Serialize type-specific point geometry
    const dt_masks_type_t base_type = form->type & ~(DT_MASKS_CLONE | DT_MASKS_NON_CLONE);

    if(base_type == DT_MASKS_GROUP)
    {
      json_builder_set_member_name(b, "children");
      json_builder_begin_array(b);
      for(GList *p = form->points; p; p = g_list_next(p))
      {
        const dt_masks_point_group_t *grpt = p->data;
        if(!grpt) continue;
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "formid");
        json_builder_add_int_value(b, grpt->formid);
        json_builder_set_member_name(b, "state");
        json_builder_add_int_value(b, grpt->state);
        json_builder_set_member_name(b, "opacity");
        json_builder_add_double_value(b, grpt->opacity);
        json_builder_end_object(b);
      }
      json_builder_end_array(b);
    }
    else if(base_type == DT_MASKS_CIRCLE)
    {
      const dt_masks_point_circle_t *pt = g_list_nth_data(form->points, 0);
      if(pt)
      {
        json_builder_set_member_name(b, "points");
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "center");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->center[0]);
        json_builder_add_double_value(b, pt->center[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "radius");
        json_builder_add_double_value(b, pt->radius);
        json_builder_set_member_name(b, "border");
        json_builder_add_double_value(b, pt->border);
        json_builder_end_object(b);
      }
    }
    else if(base_type == DT_MASKS_ELLIPSE)
    {
      const dt_masks_point_ellipse_t *pt = g_list_nth_data(form->points, 0);
      if(pt)
      {
        json_builder_set_member_name(b, "points");
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "center");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->center[0]);
        json_builder_add_double_value(b, pt->center[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "radius");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->radius[0]);
        json_builder_add_double_value(b, pt->radius[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "rotation");
        json_builder_add_double_value(b, pt->rotation);
        json_builder_set_member_name(b, "border");
        json_builder_add_double_value(b, pt->border);
        json_builder_set_member_name(b, "flags");
        json_builder_add_int_value(b, pt->flags);
        json_builder_end_object(b);
      }
    }
    else if(base_type == DT_MASKS_PATH)
    {
      json_builder_set_member_name(b, "points");
      json_builder_begin_array(b);
      for(GList *p = form->points; p; p = g_list_next(p))
      {
        const dt_masks_point_path_t *pt = p->data;
        if(!pt) continue;
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "corner");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->corner[0]);
        json_builder_add_double_value(b, pt->corner[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "ctrl1");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->ctrl1[0]);
        json_builder_add_double_value(b, pt->ctrl1[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "ctrl2");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->ctrl2[0]);
        json_builder_add_double_value(b, pt->ctrl2[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "border");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->border[0]);
        json_builder_add_double_value(b, pt->border[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "state");
        json_builder_add_int_value(b, pt->state);
        json_builder_end_object(b);
      }
      json_builder_end_array(b);
    }
    else if(base_type == DT_MASKS_BRUSH)
    {
      json_builder_set_member_name(b, "points");
      json_builder_begin_array(b);
      for(GList *p = form->points; p; p = g_list_next(p))
      {
        const dt_masks_point_brush_t *pt = p->data;
        if(!pt) continue;
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "corner");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->corner[0]);
        json_builder_add_double_value(b, pt->corner[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "ctrl1");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->ctrl1[0]);
        json_builder_add_double_value(b, pt->ctrl1[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "ctrl2");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->ctrl2[0]);
        json_builder_add_double_value(b, pt->ctrl2[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "border");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->border[0]);
        json_builder_add_double_value(b, pt->border[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "density");
        json_builder_add_double_value(b, pt->density);
        json_builder_set_member_name(b, "hardness");
        json_builder_add_double_value(b, pt->hardness);
        json_builder_set_member_name(b, "state");
        json_builder_add_int_value(b, pt->state);
        json_builder_end_object(b);
      }
      json_builder_end_array(b);
    }
    else if(base_type == DT_MASKS_GRADIENT)
    {
      const dt_masks_point_gradient_t *pt = g_list_nth_data(form->points, 0);
      if(pt)
      {
        json_builder_set_member_name(b, "points");
        json_builder_begin_object(b);
        json_builder_set_member_name(b, "anchor");
        json_builder_begin_array(b);
        json_builder_add_double_value(b, pt->anchor[0]);
        json_builder_add_double_value(b, pt->anchor[1]);
        json_builder_end_array(b);
        json_builder_set_member_name(b, "rotation");
        json_builder_add_double_value(b, pt->rotation);
        json_builder_set_member_name(b, "compression");
        json_builder_add_double_value(b, pt->compression);
        json_builder_set_member_name(b, "steepness");
        json_builder_add_double_value(b, pt->steepness);
        json_builder_set_member_name(b, "curvature");
        json_builder_add_double_value(b, pt->curvature);
        json_builder_set_member_name(b, "state");
        json_builder_add_int_value(b, pt->state);
        json_builder_end_object(b);
      }
    }

    // Transform control points through the distortion pipeline so the client
    // can render masks in output (display) coordinate space.
    {
      dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
      if(pipe && pipe->processed_width > 0 && pipe->processed_height > 0)
      {
        const float iw = (float)pipe->iwidth;
        const float ih = (float)pipe->iheight;
        const float pw = (float)pipe->processed_width;
        const float ph = (float)pipe->processed_height;

        if(base_type == DT_MASKS_CIRCLE)
        {
          const dt_masks_point_circle_t *cpt = g_list_nth_data(form->points, 0);
          if(cpt)
          {
            const float dim = MIN(iw, ih);
            const float r = cpt->radius * dim;
            // Match DT: sample count = max(10, circumference in pixels)
            const int n_samples = MAX(10, (int)(2.0f * M_PI * r));
            // center + n_samples on main circle + n_samples on border circle
            const int total = 1 + n_samples + n_samples;
            float *pts = malloc(total * 2 * sizeof(float));
            if(pts)
            {
              const float cx = cpt->center[0] * iw;
              const float cy = cpt->center[1] * ih;
              const float rb = (cpt->radius + cpt->border) * dim;
              pts[0] = cx;
              pts[1] = cy;
              for(int i = 0; i < n_samples; i++)
              {
                const float angle = 2.0f * M_PI * i / n_samples;
                const float ca = cosf(angle), sa = sinf(angle);
                pts[2 + i * 2]     = cx + r * ca;
                pts[2 + i * 2 + 1] = cy + r * sa;
                pts[2 + n_samples * 2 + i * 2]     = cx + rb * ca;
                pts[2 + n_samples * 2 + i * 2 + 1] = cy + rb * sa;
              }

              if(dt_dev_distort_transform_plus(&session->dev, pipe,
                                                0.0, DT_DEV_TRANSFORM_DIR_ALL, pts, total))
              {
                json_builder_set_member_name(b, "transformed");
                json_builder_begin_object(b);
                json_builder_set_member_name(b, "center");
                json_builder_begin_array(b);
                json_builder_add_double_value(b, pts[0] / pw);
                json_builder_add_double_value(b, pts[1] / ph);
                json_builder_end_array(b);
                json_builder_set_member_name(b, "main_polyline");
                json_builder_begin_array(b);
                for(int i = 0; i < n_samples; i++)
                {
                  json_builder_add_double_value(b, pts[2 + i * 2] / pw);
                  json_builder_add_double_value(b, pts[2 + i * 2 + 1] / ph);
                }
                json_builder_end_array(b);
                json_builder_set_member_name(b, "border_polyline");
                json_builder_begin_array(b);
                for(int i = 0; i < n_samples; i++)
                {
                  json_builder_add_double_value(b, pts[2 + n_samples * 2 + i * 2] / pw);
                  json_builder_add_double_value(b, pts[2 + n_samples * 2 + i * 2 + 1] / ph);
                }
                json_builder_end_array(b);
                // Also emit radius/border for handle positioning
                const float odim = MIN(pw, ph);
                const float dx_r = pts[2] - pts[0], dy_r = pts[3] - pts[1];
                const float r_out = sqrtf(dx_r * dx_r + dy_r * dy_r);
                const float dx_b = pts[2 + n_samples * 2] - pts[0], dy_b = pts[2 + n_samples * 2 + 1] - pts[1];
                const float rb_out = sqrtf(dx_b * dx_b + dy_b * dy_b);
                json_builder_set_member_name(b, "radius");
                json_builder_add_double_value(b, r_out / odim);
                json_builder_set_member_name(b, "border");
                json_builder_add_double_value(b, (rb_out - r_out) / odim);
                json_builder_end_object(b);
              }
              free(pts);
            }
          }
        }
        else if(base_type == DT_MASKS_ELLIPSE)
        {
          const dt_masks_point_ellipse_t *ept = g_list_nth_data(form->points, 0);
          if(ept)
          {
            const float dim = MIN(iw, ih);
            // Match DT's _points_to_transform: swap axes so a >= b
            const float v1 = ept->rotation * (M_PI / 180.0f);
            const float v2 = v1 - M_PI / 2.0f;
            float a, b_ax, v;
            if(ept->radius[0] >= ept->radius[1])
            { a = ept->radius[0] * dim; b_ax = ept->radius[1] * dim; v = v1; }
            else
            { a = ept->radius[1] * dim; b_ax = ept->radius[0] * dim; v = v2; }

            const float sinv = sinf(v), cosv = cosf(v);

            // Border radii (match DT's proportional vs absolute)
            const int prop = ept->flags & 1; // DT_MASKS_ELLIPSE_PROPORTIONAL
            float ab, bb;
            if(prop)
            { ab = a * (1.0f + ept->border); bb = b_ax * (1.0f + ept->border); }
            else
            { ab = a + ept->border * dim; bb = b_ax + ept->border * dim; }

            // Sample count: Ramanujan approximation (match DT)
            const float lambda = (a - b_ax) / (a + b_ax + 1e-10f);
            const int n_el = MAX(100, (int)((M_PI * (a + b_ax)
                * (1.0f + (3.0f * lambda * lambda)
                   / (10.0f + sqrtf(4.0f - 3.0f * lambda * lambda)))) / 10));

            // center + n_el main ellipse pts + n_el border ellipse pts
            const int total = 1 + n_el + n_el;
            float *pts = malloc(total * 2 * sizeof(float));
            if(pts)
            {
              const float cx = ept->center[0] * iw;
              const float cy = ept->center[1] * ih;
              pts[0] = cx;
              pts[1] = cy;
              for(int i = 0; i < n_el; i++)
              {
                const float alpha = i * 2.0f * M_PI / (float)n_el;
                const float cos_a = cosf(alpha), sin_a = sinf(alpha);
                // Main ellipse contour
                pts[2 + i * 2]     = cx + a * cos_a * cosv - b_ax * sin_a * sinv;
                pts[2 + i * 2 + 1] = cy + a * cos_a * sinv + b_ax * sin_a * cosv;
                // Border ellipse contour
                pts[2 + n_el * 2 + i * 2]     = cx + ab * cos_a * cosv - bb * sin_a * sinv;
                pts[2 + n_el * 2 + i * 2 + 1] = cy + ab * cos_a * sinv + bb * sin_a * cosv;
              }

              if(dt_dev_distort_transform_plus(&session->dev, pipe,
                                                0.0, DT_DEV_TRANSFORM_DIR_ALL, pts, total))
              {
                json_builder_set_member_name(b, "transformed");
                json_builder_begin_object(b);
                json_builder_set_member_name(b, "center");
                json_builder_begin_array(b);
                json_builder_add_double_value(b, pts[0] / pw);
                json_builder_add_double_value(b, pts[1] / ph);
                json_builder_end_array(b);
                json_builder_set_member_name(b, "main_polyline");
                json_builder_begin_array(b);
                for(int i = 0; i < n_el; i++)
                {
                  json_builder_add_double_value(b, pts[2 + i * 2] / pw);
                  json_builder_add_double_value(b, pts[2 + i * 2 + 1] / ph);
                }
                json_builder_end_array(b);
                json_builder_set_member_name(b, "border_polyline");
                json_builder_begin_array(b);
                for(int i = 0; i < n_el; i++)
                {
                  json_builder_add_double_value(b, pts[2 + n_el * 2 + i * 2] / pw);
                  json_builder_add_double_value(b, pts[2 + n_el * 2 + i * 2 + 1] / ph);
                }
                json_builder_end_array(b);
                json_builder_end_object(b);
              }
              free(pts);
            }
          }
        }
        else if(base_type == DT_MASKS_PATH)
        {
          const int n = g_list_length(form->points);
          if(n > 0)
          {
            // Control points + border polyline generated in input space
            const int border_samples = 20;
            const int polyline_per_seg = border_samples + 1;
            const int border_pts = n * polyline_per_seg;
            const int ctrl_pts = n * 3;
            const int total_pts = ctrl_pts + border_pts;
            float *pts = malloc(total_pts * 2 * sizeof(float));
            if(pts)
            {
              int idx = 0;
              for(GList *p = form->points; p; p = g_list_next(p))
              {
                const dt_masks_point_path_t *ppt = p->data;
                if(!ppt) { idx += 6; continue; }
                pts[idx++] = ppt->corner[0] * iw;
                pts[idx++] = ppt->corner[1] * ih;
                pts[idx++] = ppt->ctrl1[0] * iw;
                pts[idx++] = ppt->ctrl1[1] * ih;
                pts[idx++] = ppt->ctrl2[0] * iw;
                pts[idx++] = ppt->ctrl2[1] * ih;
              }

              // Winding direction
              float area = 0;
              for(GList *p = form->points; p; p = g_list_next(p))
              {
                const dt_masks_point_path_t *curr = p->data;
                const dt_masks_point_path_t *next = g_list_next(p)
                  ? g_list_next(p)->data : form->points->data;
                area += (next->corner[0] - curr->corner[0])
                      * (next->corner[1] + curr->corner[1]);
              }
              const float cw = area < 0 ? 1.0f : -1.0f;
              const float idim = MIN(iw, ih);

              // Sample border polyline in input space
              idx = ctrl_pts * 2;
              for(GList *p = form->points; p; p = g_list_next(p))
              {
                const dt_masks_point_path_t *pt1 = p->data;
                const dt_masks_point_path_t *pt2 = g_list_next(p)
                  ? g_list_next(p)->data : form->points->data;
                const float p0x = pt1->corner[0]*iw, p0y = pt1->corner[1]*ih;
                const float p1x = pt1->ctrl2[0]*iw,  p1y = pt1->ctrl2[1]*ih;
                const float p2x = pt2->ctrl1[0]*iw,  p2y = pt2->ctrl1[1]*ih;
                const float p3x = pt2->corner[0]*iw, p3y = pt2->corner[1]*ih;
                const float rs = cw * pt1->border[1] * idim;
                const float re = cw * pt2->border[0] * idim;

                for(int s = 0; s <= border_samples; s++)
                {
                  const float t = (float)s / border_samples;
                  const float rad = rs + (re - rs) * t;
                  const float ti = 1.0f - t;
                  const float ti2 = ti*ti, ti3 = ti2*ti;
                  const float t2 = t*t, t3 = t2*t;
                  const float cx = ti3*p0x + 3*ti2*t*p1x + 3*ti*t2*p2x + t3*p3x;
                  const float cy = ti3*p0y + 3*ti2*t*p1y + 3*ti*t2*p2y + t3*p3y;
                  const float a3 = 3*ti*ti, bv = 3*(ti*ti - 2*t*ti);
                  const float cv = 3*(2*t*ti - t*t), d3 = 3*t*t;
                  const float dx = -p0x*a3 + p1x*bv + p2x*cv + p3x*d3;
                  const float dy = -p0y*a3 + p1y*bv + p2y*cv + p3y*d3;
                  const float len = sqrtf(dx*dx + dy*dy);
                  if(len > 1e-10f)
                  { pts[idx++] = cx + rad*dy/len; pts[idx++] = cy - rad*dx/len; }
                  else
                  { pts[idx++] = cx; pts[idx++] = cy; }
                }
              }

              if(dt_dev_distort_transform_plus(&session->dev, pipe,
                                                0.0, DT_DEV_TRANSFORM_DIR_ALL, pts, total_pts))
              {
                json_builder_set_member_name(b, "transformed");
                json_builder_begin_object(b);
                json_builder_set_member_name(b, "controls");
                json_builder_begin_array(b);
                idx = 0;
                GList *pp = form->points;
                for(int i = 0; i < n; i++)
                {
                  const dt_masks_point_path_t *ppt = pp ? pp->data : NULL;
                  json_builder_begin_object(b);
                  json_builder_set_member_name(b, "corner");
                  json_builder_begin_array(b);
                  json_builder_add_double_value(b, pts[idx] / pw);
                  json_builder_add_double_value(b, pts[idx + 1] / ph);
                  json_builder_end_array(b);
                  json_builder_set_member_name(b, "ctrl1");
                  json_builder_begin_array(b);
                  json_builder_add_double_value(b, pts[idx + 2] / pw);
                  json_builder_add_double_value(b, pts[idx + 3] / ph);
                  json_builder_end_array(b);
                  json_builder_set_member_name(b, "ctrl2");
                  json_builder_begin_array(b);
                  json_builder_add_double_value(b, pts[idx + 4] / pw);
                  json_builder_add_double_value(b, pts[idx + 5] / ph);
                  json_builder_end_array(b);
                  json_builder_set_member_name(b, "border");
                  json_builder_begin_array(b);
                  json_builder_add_double_value(b, ppt ? ppt->border[0] : 0.0);
                  json_builder_add_double_value(b, ppt ? ppt->border[1] : 0.0);
                  json_builder_end_array(b);
                  json_builder_end_object(b);
                  idx += 6;
                  if(pp) pp = g_list_next(pp);
                }
                json_builder_end_array(b);

                json_builder_set_member_name(b, "border_polyline");
                json_builder_begin_array(b);
                idx = ctrl_pts * 2;
                for(int i = 0; i < border_pts; i++)
                {
                  json_builder_add_double_value(b, pts[idx] / pw);
                  json_builder_add_double_value(b, pts[idx + 1] / ph);
                  idx += 2;
                }
                json_builder_end_array(b);
                json_builder_end_object(b);
              }
              free(pts);
            }
          }
        }
        else if(base_type == DT_MASKS_BRUSH)
        {
          const int n = g_list_length(form->points);
          if(n > 1)
          {
            const int border_samples = 40;
            const int polyline_per_seg = border_samples + 1;
            const int n_segs = n - 1;
            const int border_pts_per_side = n_segs * polyline_per_seg;
            const int ctrl_pts = n * 3;
            const int total_pts = ctrl_pts + border_pts_per_side * 2;
            float *pts = malloc(total_pts * 2 * sizeof(float));
            if(pts)
            {
              int idx = 0;
              for(GList *p = form->points; p; p = g_list_next(p))
              {
                const dt_masks_point_brush_t *bpt = p->data;
                if(!bpt) { idx += 6; continue; }
                pts[idx++] = bpt->corner[0] * iw;
                pts[idx++] = bpt->corner[1] * ih;
                pts[idx++] = bpt->ctrl1[0] * iw;
                pts[idx++] = bpt->ctrl1[1] * ih;
                pts[idx++] = bpt->ctrl2[0] * iw;
                pts[idx++] = bpt->ctrl2[1] * ih;
              }

              const float idim = MIN(iw, ih);
              for(int side = 0; side < 2; side++)
              {
                const float sign = (side == 0) ? 1.0f : -1.0f;
                GList *p = form->points;
                for(int k = 0; k < n_segs; k++)
                {
                  const dt_masks_point_brush_t *pt1 = p->data;
                  const dt_masks_point_brush_t *pt2 = g_list_next(p)->data;
                  const float p0x = pt1->corner[0]*iw, p0y = pt1->corner[1]*ih;
                  const float p1x = pt1->ctrl2[0]*iw,  p1y = pt1->ctrl2[1]*ih;
                  const float p2x = pt2->ctrl1[0]*iw,  p2y = pt2->ctrl1[1]*ih;
                  const float p3x = pt2->corner[0]*iw, p3y = pt2->corner[1]*ih;
                  const float rs = sign * pt1->border[1] * idim;
                  const float re = sign * pt2->border[0] * idim;

                  for(int s = 0; s <= border_samples; s++)
                  {
                    const float t = (float)s / border_samples;
                    const float rad = rs + (re - rs) * t;
                    const float ti = 1.0f - t;
                    const float ti2 = ti*ti, ti3 = ti2*ti;
                    const float t2 = t*t, t3 = t2*t;
                    const float cx = ti3*p0x + 3*ti2*t*p1x + 3*ti*t2*p2x + t3*p3x;
                    const float cy = ti3*p0y + 3*ti2*t*p1y + 3*ti*t2*p2y + t3*p3y;
                    const float a3 = 3*ti*ti, bv = 3*(ti*ti - 2*t*ti);
                    const float cv = 3*(2*t*ti - t*t), d3 = 3*t*t;
                    const float dx = -p0x*a3 + p1x*bv + p2x*cv + p3x*d3;
                    const float dy = -p0y*a3 + p1y*bv + p2y*cv + p3y*d3;
                    const float len = sqrtf(dx*dx + dy*dy);
                    if(len > 1e-10f)
                    { pts[idx++] = cx + rad*dy/len; pts[idx++] = cy - rad*dx/len; }
                    else
                    { pts[idx++] = cx; pts[idx++] = cy; }
                  }
                  p = g_list_next(p);
                }
              }

              if(dt_dev_distort_transform_plus(&session->dev, pipe,
                                                0.0, DT_DEV_TRANSFORM_DIR_ALL, pts, total_pts))
              {
                json_builder_set_member_name(b, "transformed");
                json_builder_begin_object(b);
                json_builder_set_member_name(b, "controls");
                json_builder_begin_array(b);
                idx = 0;
                for(GList *p = form->points; p; p = g_list_next(p))
                {
                  const dt_masks_point_brush_t *bpt = p->data;
                  json_builder_begin_object(b);
                  json_builder_set_member_name(b, "corner");
                  json_builder_begin_array(b);
                  json_builder_add_double_value(b, pts[idx] / pw);
                  json_builder_add_double_value(b, pts[idx + 1] / ph);
                  json_builder_end_array(b);
                  json_builder_set_member_name(b, "ctrl1");
                  json_builder_begin_array(b);
                  json_builder_add_double_value(b, pts[idx + 2] / pw);
                  json_builder_add_double_value(b, pts[idx + 3] / ph);
                  json_builder_end_array(b);
                  json_builder_set_member_name(b, "ctrl2");
                  json_builder_begin_array(b);
                  json_builder_add_double_value(b, pts[idx + 4] / pw);
                  json_builder_add_double_value(b, pts[idx + 5] / ph);
                  json_builder_end_array(b);
                  if(bpt)
                  {
                    json_builder_set_member_name(b, "border");
                    json_builder_begin_array(b);
                    json_builder_add_double_value(b, bpt->border[0]);
                    json_builder_add_double_value(b, bpt->border[1]);
                    json_builder_end_array(b);
                    json_builder_set_member_name(b, "density");
                    json_builder_add_double_value(b, bpt->density);
                    json_builder_set_member_name(b, "hardness");
                    json_builder_add_double_value(b, bpt->hardness);
                    json_builder_set_member_name(b, "state");
                    json_builder_add_int_value(b, bpt->state);
                  }
                  json_builder_end_object(b);
                  idx += 6;
                }
                json_builder_end_array(b);

                idx = ctrl_pts * 2;
                for(int side = 0; side < 2; side++)
                {
                  json_builder_set_member_name(b, side == 0 ? "border_polyline1" : "border_polyline2");
                  json_builder_begin_array(b);
                  for(int i = 0; i < border_pts_per_side; i++)
                  {
                    json_builder_add_double_value(b, pts[idx] / pw);
                    json_builder_add_double_value(b, pts[idx + 1] / ph);
                    idx += 2;
                  }
                  json_builder_end_array(b);
                }
                json_builder_end_object(b);
              }
              free(pts);
            }
          }
        }
        else if(base_type == DT_MASKS_GRADIENT)
        {
          const dt_masks_point_gradient_t *gpt = g_list_nth_data(form->points, 0);
          if(gpt)
          {
            // Match DT's _gradient_get_points: sample line + two border lines
            const float scale = sqrtf(iw * iw + ih * ih);
            const float v = -(gpt->rotation) * (M_PI / 180.0f);
            const float cosv = cosf(v);
            const float sinv = sinf(v);
            const float curv = gpt->curvature;
            const float comp = gpt->compression;
            const float ax = gpt->anchor[0], ay = gpt->anchor[1];

            // Line sample count (match DT)
            const int n_line = (int)scale;
            const float xstart = fabsf(curv) > 1.0f ? -sqrtf(1.0f / fabsf(curv)) : -1.0f;
            const float xdelta = -2.0f * xstart / (float)(n_line > 1 ? n_line - 1 : 1);

            // 3 lines: main + border1 + border2, plus anchor point
            // anchor + ref_point + n_line * 3
            const int total = 2 + n_line * 3;
            float *pts = malloc(total * 2 * sizeof(float));
            if(pts)
            {
              // Anchor + reference point for rotation recovery
              const float ref_dist = 0.1f * MIN(iw, ih);
              const float rot_rad = gpt->rotation * (M_PI / 180.0f);
              pts[0] = ax * iw;
              pts[1] = ay * ih;
              pts[2] = pts[0] + ref_dist * sinf(rot_rad);
              pts[3] = pts[1] + ref_dist * cosf(rot_rad);

              // Border offset directions: -(rotation ± 90°)
              const float v1 = -(gpt->rotation - 90.0f) * (M_PI / 180.0f);
              const float v2 = -(gpt->rotation + 90.0f) * (M_PI / 180.0f);
              // Offset anchor positions for border lines
              const float b1x = (ax * iw + comp * scale * cosf(v1)) / iw;
              const float b1y = (ay * ih + comp * scale * sinf(v1)) / ih;
              const float b2x = (ax * iw + comp * scale * cosf(v2)) / iw;
              const float b2y = (ay * ih + comp * scale * sinf(v2)) / ih;

              int idx = 4; // after anchor + ref point
              // Generate 3 lines: main, border1, border2
              const float centers[3][2] = {
                { ax, ay }, { b1x, b1y }, { b2x, b2y }
              };
              for(int line = 0; line < 3; line++)
              {
                const float cx = centers[line][0];
                const float cy = centers[line][1];
                for(int i = 0; i < n_line; i++)
                {
                  const float xi = xstart + i * xdelta;
                  const float yi = curv * xi * xi;
                  const float xii = (cosv * xi + sinv * yi) * scale;
                  const float yii = (sinv * xi - cosv * yi) * scale;
                  pts[idx++] = cx * iw + xii;
                  pts[idx++] = cy * ih + yii;
                }
              }

              if(dt_dev_distort_transform_plus(&session->dev, pipe,
                                                0.0, DT_DEV_TRANSFORM_DIR_ALL, pts, total))
              {
                json_builder_set_member_name(b, "transformed");
                json_builder_begin_object(b);
                // Anchor
                json_builder_set_member_name(b, "anchor");
                json_builder_begin_array(b);
                json_builder_add_double_value(b, pts[0] / pw);
                json_builder_add_double_value(b, pts[1] / ph);
                json_builder_end_array(b);
                // Recover rotation from transformed ref point
                const float odim = MIN(pw, ph);
                const float rdx = (pts[2] - pts[0]) / odim;
                const float rdy = (pts[3] - pts[1]) / odim;
                json_builder_set_member_name(b, "rotation");
                json_builder_add_double_value(b, atan2f(rdx, rdy) * (180.0f / M_PI));
                // Main line polyline
                json_builder_set_member_name(b, "main_polyline");
                json_builder_begin_array(b);
                idx = 4;
                for(int i = 0; i < n_line; i++)
                {
                  json_builder_add_double_value(b, pts[idx] / pw);
                  json_builder_add_double_value(b, pts[idx + 1] / ph);
                  idx += 2;
                }
                json_builder_end_array(b);
                // Border line 1
                json_builder_set_member_name(b, "border_polyline1");
                json_builder_begin_array(b);
                for(int i = 0; i < n_line; i++)
                {
                  json_builder_add_double_value(b, pts[idx] / pw);
                  json_builder_add_double_value(b, pts[idx + 1] / ph);
                  idx += 2;
                }
                json_builder_end_array(b);
                // Border line 2
                json_builder_set_member_name(b, "border_polyline2");
                json_builder_begin_array(b);
                for(int i = 0; i < n_line; i++)
                {
                  json_builder_add_double_value(b, pts[idx] / pw);
                  json_builder_add_double_value(b, pts[idx + 1] / ph);
                  idx += 2;
                }
                json_builder_end_array(b);
                // Pass through other params
                json_builder_set_member_name(b, "compression");
                json_builder_add_double_value(b, comp);
                json_builder_set_member_name(b, "steepness");
                json_builder_add_double_value(b, gpt->steepness);
                json_builder_set_member_name(b, "curvature");
                json_builder_add_double_value(b, curv);
                json_builder_set_member_name(b, "state");
                json_builder_add_int_value(b, gpt->state);
                json_builder_end_object(b);
              }
              free(pts);
            }
          }
        }
      }
    }

    json_builder_end_object(b);
  }

  json_builder_end_array(b);

  json_builder_set_member_name(b, "usage");
  json_builder_begin_array(b);

  for(GList *m = session->dev.iop; m; m = g_list_next(m))
  {
    const dt_iop_module_t *module = m->data;
    if(!module || !module->blend_params) continue;
    if(!dt_is_valid_maskid(module->blend_params->mask_id)) continue;

    json_builder_begin_object(b);
    json_builder_set_member_name(b, "mask_id");
    json_builder_add_int_value(b, module->blend_params->mask_id);
    json_builder_set_member_name(b, "op");
    json_builder_add_string_value(b, module->op);
    json_builder_set_member_name(b, "instance");
    json_builder_add_int_value(b, module->multi_priority);
    if(module->name)
    {
      json_builder_set_member_name(b, "module_name");
      json_builder_add_string_value(b, module->name());
    }
    json_builder_end_object(b);
  }

  json_builder_end_array(b);
  json_builder_end_object(b);

  JsonNode *masks_result = json_builder_get_root(b);
  char *masks_resp = dt_server_make_response(req->id, masks_result);
  json_node_unref(masks_result);
  g_object_unref(b);
  return masks_resp;
}

char *dt_server_develop_rename_mask(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "formid")
     || !json_object_has_member(req->params, "name"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, formid or name parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  const dt_mask_id_t formid = (dt_mask_id_t)json_object_get_int_member(req->params, "formid");
  const char *name = json_object_get_string_member(req->params, "name");
  if(!name)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "name must be a string");

  dt_masks_form_t *form = dt_masks_get_from_id_ext(session->dev.forms, formid);
  if(!form)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Mask form not found");

  g_strlcpy(form->name, name, sizeof(form->name));
  dt_dev_add_masks_history_item_ext(&session->dev, NULL, TRUE, FALSE);
  dt_dev_write_history(&session->dev);

  return dt_server_make_response(req->id, json_node_new(JSON_NODE_NULL));
}

char *dt_server_develop_delete_mask(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "formid"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or formid parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  const dt_mask_id_t formid = (dt_mask_id_t)json_object_get_int_member(req->params, "formid");

  dt_masks_form_t *form = dt_masks_get_from_id_ext(session->dev.forms, formid);
  if(!form)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Mask form not found");

  // Remove this form from any module groups that reference it
  for(GList *m = session->dev.iop; m; m = g_list_next(m))
  {
    dt_iop_module_t *module = m->data;
    if(!module || !module->blend_params) continue;
    if(!(module->flags() & IOP_FLAGS_SUPPORTS_BLENDING)) continue;

    // Check if this is the module's base mask group
    if(module->blend_params->mask_id == formid)
    {
      module->blend_params->mask_id = NO_MASKID;
      dt_dev_add_history_item_ext(&session->dev, module, module->enabled, TRUE);
      continue;
    }

    // Check if this form is inside the module's mask group
    dt_masks_form_t *grp = dt_masks_get_from_id_ext(session->dev.forms,
                                                     module->blend_params->mask_id);
    if(!grp || !(grp->type & DT_MASKS_GROUP)) continue;

    for(GList *p = grp->points; p; p = g_list_next(p))
    {
      dt_masks_point_group_t *grpt = p->data;
      if(grpt->formid == formid)
      {
        grp->points = g_list_remove(grp->points, grpt);
        free(grpt);
        dt_dev_add_history_item_ext(&session->dev, module, module->enabled, TRUE);
        break;
      }
    }
  }

  // Remove from dev->forms list
  for(GList *l = session->dev.forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    if(f->formid == formid)
    {
      session->dev.forms = g_list_remove(session->dev.forms, f);
      dt_masks_free_form(f);
      break;
    }
  }

  dt_dev_add_masks_history_item_ext(&session->dev, NULL, TRUE, FALSE);
  dt_dev_write_history(&session->dev);

  return dt_server_make_response(req->id, json_node_new(JSON_NODE_NULL));
}

// ─── Distortion grid ──────────────────────────────────────────────────────────

#define DISTORTION_GRID_SIZE 64

char *dt_server_develop_get_distortion_grid(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Missing session_id parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;
  if(!pipe || pipe->processed_width <= 0 || pipe->processed_height <= 0)
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                "Pipeline not processed yet");

  const int gw = DISTORTION_GRID_SIZE;
  const int gh = DISTORTION_GRID_SIZE;
  const int total = gw * gh;
  const float iw = (float)pipe->iwidth;
  const float ih = (float)pipe->iheight;
  const float pw = (float)pipe->processed_width;
  const float ph = (float)pipe->processed_height;

  // Allocate points for forward transform (raw → output)
  float *fwd_pts = malloc(total * 2 * sizeof(float));
  if(!fwd_pts)
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "Out of memory");

  // Allocate points for inverse transform (output → raw)
  float *inv_pts = malloc(total * 2 * sizeof(float));
  if(!inv_pts)
  {
    free(fwd_pts);
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "Out of memory");
  }

  // Fill forward grid: regular grid in raw normalized space → pixel coords
  for(int j = 0; j < gh; j++)
  {
    for(int i = 0; i < gw; i++)
    {
      const int idx = (j * gw + i) * 2;
      const float nx = (float)i / (float)(gw - 1);
      const float ny = (float)j / (float)(gh - 1);
      fwd_pts[idx]     = nx * iw;
      fwd_pts[idx + 1] = ny * ih;
    }
  }

  // Fill inverse grid: regular grid in output normalized space → pixel coords
  for(int j = 0; j < gh; j++)
  {
    for(int i = 0; i < gw; i++)
    {
      const int idx = (j * gw + i) * 2;
      const float nx = (float)i / (float)(gw - 1);
      const float ny = (float)j / (float)(gh - 1);
      inv_pts[idx]     = nx * pw;
      inv_pts[idx + 1] = ny * ph;
    }
  }

  // Transform: forward (raw space → output space)
  const gboolean fwd_ok = dt_dev_distort_transform_plus(
      &session->dev, pipe, 0.0, DT_DEV_TRANSFORM_DIR_ALL, fwd_pts, total);

  // Transform: inverse (output space → raw space)
  const gboolean inv_ok = dt_dev_distort_backtransform_plus(
      &session->dev, pipe, 0.0, DT_DEV_TRANSFORM_DIR_ALL, inv_pts, total);

  // Build JSON response
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "width");
  json_builder_add_int_value(b, gw);
  json_builder_set_member_name(b, "height");
  json_builder_add_int_value(b, gh);
  // Pipeline hash — client uses this to detect when the grid is stale
  json_builder_set_member_name(b, "pipe_hash");
  json_builder_add_int_value(b, (gint64)pipe->backbuf_hash);
  json_builder_set_member_name(b, "iwidth");
  json_builder_add_int_value(b, pipe->iwidth);
  json_builder_set_member_name(b, "iheight");
  json_builder_add_int_value(b, pipe->iheight);
  json_builder_set_member_name(b, "processed_width");
  json_builder_add_int_value(b, pipe->processed_width);
  json_builder_set_member_name(b, "processed_height");
  json_builder_add_int_value(b, pipe->processed_height);

  // Forward grid: output in normalized output space [0,1]
  json_builder_set_member_name(b, "forward");
  json_builder_begin_array(b);
  if(fwd_ok)
  {
    for(int k = 0; k < total; k++)
    {
      json_builder_add_double_value(b, (double)(fwd_pts[k * 2] / pw));
      json_builder_add_double_value(b, (double)(fwd_pts[k * 2 + 1] / ph));
    }
  }
  json_builder_end_array(b);

  // Inverse grid: output in normalized raw space [0,1]
  json_builder_set_member_name(b, "inverse");
  json_builder_begin_array(b);
  if(inv_ok)
  {
    for(int k = 0; k < total; k++)
    {
      json_builder_add_double_value(b, (double)(inv_pts[k * 2] / iw));
      json_builder_add_double_value(b, (double)(inv_pts[k * 2 + 1] / ih));
    }
  }
  json_builder_end_array(b);

  json_builder_end_object(b);

  free(fwd_pts);
  free(inv_pts);

  JsonNode *root = json_builder_get_root(b);
  g_object_unref(b);
  return dt_server_make_response(req->id, root);
}

// Ensure a form ID is unique within the session's forms list
static void _server_check_mask_id(dt_develop_t *dev, dt_masks_form_t *form)
{
  dt_mask_id_t nid = 100;
  for(GList *forms = dev->forms; forms; )
  {
    const dt_masks_form_t *ff = forms->data;
    if(ff->formid == form->formid)
    {
      form->formid = nid++;
      forms = dev->forms; // restart scan
    }
    else
      forms = g_list_next(forms);
  }
}

// Auto-generate a unique name like "circle #1", "ellipse #2"
static void _server_set_form_name(dt_develop_t *dev, dt_masks_form_t *form)
{
  guint nb = 0;
  for(GList *l = dev->forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *f = l->data;
    if(f->type == form->type) nb++;
  }
  gboolean exist;
  do
  {
    exist = FALSE;
    nb++;
    if(form->functions && form->functions->set_form_name)
      form->functions->set_form_name(form, nb);
    for(GList *l = dev->forms; l; l = g_list_next(l))
    {
      const dt_masks_form_t *f = l->data;
      if(!strcmp(f->name, form->name)) { exist = TRUE; break; }
    }
  } while(exist);
}

char *dt_server_develop_create_mask(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "type"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or type parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  const char *type_str = json_object_get_string_member(req->params, "type");
  if(!type_str)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "type must be a string");

  dt_masks_type_t mask_type = DT_MASKS_NONE;
  if(!strcmp(type_str, "circle")) mask_type = DT_MASKS_CIRCLE;
  else if(!strcmp(type_str, "ellipse")) mask_type = DT_MASKS_ELLIPSE;
  else if(!strcmp(type_str, "path")) mask_type = DT_MASKS_PATH;
  else if(!strcmp(type_str, "brush")) mask_type = DT_MASKS_BRUSH;
  else if(!strcmp(type_str, "gradient")) mask_type = DT_MASKS_GRADIENT;
  else
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Unknown mask type");

  // Create the form
  dt_masks_form_t *form = dt_masks_create(mask_type);
  if(!form)
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "Failed to create mask form");

  _server_check_mask_id(&session->dev, form);
  _server_set_form_name(&session->dev, form);

  // Set geometry from params
  if(mask_type == DT_MASKS_CIRCLE)
  {
    dt_masks_point_circle_t *circle = calloc(1, sizeof(dt_masks_point_circle_t));
    if(json_object_has_member(req->params, "center"))
    {
      JsonArray *c = json_object_get_array_member(req->params, "center");
      circle->center[0] = (float)json_array_get_double_element(c, 0);
      circle->center[1] = (float)json_array_get_double_element(c, 1);
    }
    else
    {
      circle->center[0] = 0.5f;
      circle->center[1] = 0.5f;
    }
    circle->radius = json_object_has_member(req->params, "radius")
      ? (float)json_object_get_double_member(req->params, "radius") : 0.1f;
    circle->border = json_object_has_member(req->params, "border")
      ? (float)json_object_get_double_member(req->params, "border") : 0.05f;
    form->points = g_list_append(form->points, circle);
  }
  else if(mask_type == DT_MASKS_ELLIPSE)
  {
    dt_masks_point_ellipse_t *ellipse = calloc(1, sizeof(dt_masks_point_ellipse_t));
    if(json_object_has_member(req->params, "center"))
    {
      JsonArray *c = json_object_get_array_member(req->params, "center");
      ellipse->center[0] = (float)json_array_get_double_element(c, 0);
      ellipse->center[1] = (float)json_array_get_double_element(c, 1);
    }
    else
    {
      ellipse->center[0] = 0.5f;
      ellipse->center[1] = 0.5f;
    }
    if(json_object_has_member(req->params, "radius"))
    {
      JsonArray *r = json_object_get_array_member(req->params, "radius");
      ellipse->radius[0] = (float)json_array_get_double_element(r, 0);
      ellipse->radius[1] = (float)json_array_get_double_element(r, 1);
    }
    else
    {
      ellipse->radius[0] = 0.1f;
      ellipse->radius[1] = 0.1f;
    }
    ellipse->rotation = json_object_has_member(req->params, "rotation")
      ? (float)json_object_get_double_member(req->params, "rotation") : 0.0f;
    ellipse->border = json_object_has_member(req->params, "border")
      ? (float)json_object_get_double_member(req->params, "border") : 0.05f;
    ellipse->flags = json_object_has_member(req->params, "flags")
      ? (dt_masks_ellipse_flags_t)json_object_get_int_member(req->params, "flags")
      : DT_MASKS_ELLIPSE_EQUIDISTANT;
    form->points = g_list_append(form->points, ellipse);
  }
  else if(mask_type == DT_MASKS_GRADIENT)
  {
    dt_masks_point_gradient_t *gradient = calloc(1, sizeof(dt_masks_point_gradient_t));
    if(json_object_has_member(req->params, "anchor"))
    {
      JsonArray *a = json_object_get_array_member(req->params, "anchor");
      gradient->anchor[0] = (float)json_array_get_double_element(a, 0);
      gradient->anchor[1] = (float)json_array_get_double_element(a, 1);
    }
    else
    {
      gradient->anchor[0] = 0.5f;
      gradient->anchor[1] = 0.5f;
    }
    gradient->rotation = json_object_has_member(req->params, "rotation")
      ? (float)json_object_get_double_member(req->params, "rotation") : 0.0f;
    gradient->compression = json_object_has_member(req->params, "compression")
      ? (float)json_object_get_double_member(req->params, "compression") : 0.5f;
    gradient->steepness = json_object_has_member(req->params, "steepness")
      ? (float)json_object_get_double_member(req->params, "steepness") : 0.0f;
    gradient->curvature = json_object_has_member(req->params, "curvature")
      ? (float)json_object_get_double_member(req->params, "curvature") : 0.0f;
    gradient->state = json_object_has_member(req->params, "state")
      ? (dt_masks_gradient_states_t)json_object_get_int_member(req->params, "state")
      : DT_MASKS_GRADIENT_STATE_SIGMOIDAL;
    form->points = g_list_append(form->points, gradient);
  }

  // Register form
  session->dev.forms = g_list_append(session->dev.forms, form);
  dt_dev_add_masks_history_item_ext(&session->dev, NULL, TRUE, FALSE);

  // If module specified, assign to its mask group
  if(json_object_has_member(req->params, "op"))
  {
    const char *op = json_object_get_string_member(req->params, "op");
    const int instance = json_object_has_member(req->params, "instance")
      ? (int)json_object_get_int_member(req->params, "instance") : 0;
    dt_iop_module_t *module = _find_module(session, op, instance);
    if(module && (module->flags() & IOP_FLAGS_SUPPORTS_BLENDING) && module->blend_params)
    {
      // Get or create group
      dt_masks_form_t *grp = dt_masks_get_from_id_ext(session->dev.forms,
                                                       module->blend_params->mask_id);
      if(!grp || !(grp->type & DT_MASKS_GROUP))
      {
        grp = dt_masks_create(DT_MASKS_GROUP);
        _server_check_mask_id(&session->dev, grp);
        snprintf(grp->name, sizeof(grp->name), "group %s", op);
        session->dev.forms = g_list_append(session->dev.forms, grp);
        module->blend_params->mask_id = grp->formid;
      }

      // Add form to group
      dt_masks_point_group_t *grpt = malloc(sizeof(dt_masks_point_group_t));
      grpt->formid = form->formid;
      grpt->parentid = grp->formid;
      grpt->state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE;
      if(grp->points)
        grpt->state |= DT_MASKS_STATE_UNION;
      grpt->opacity = 1.0f;
      grp->points = g_list_append(grp->points, grpt);

      // Enable drawn mask mode if not already
      if(!(module->blend_params->mask_mode & DEVELOP_MASK_MASK))
        module->blend_params->mask_mode |= DEVELOP_MASK_MASK;

      dt_iop_commit_blend_params(module, module->blend_params);
      dt_dev_add_history_item_ext(&session->dev, module, module->enabled, TRUE);
      dt_dev_add_masks_history_item_ext(&session->dev, NULL, TRUE, FALSE);
    }
  }

  dt_dev_write_history(&session->dev);

  // Return the new formid
  JsonObject *result = json_object_new();
  json_object_set_int_member(result, "formid", form->formid);
  json_object_set_string_member(result, "name", form->name);
  JsonNode *node = json_node_new(JSON_NODE_OBJECT);
  json_node_set_object(node, result);
  json_object_unref(result);
  return dt_server_make_response(req->id, node);
}

char *dt_server_develop_update_mask(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "formid"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id or formid parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  const dt_mask_id_t formid = (dt_mask_id_t)json_object_get_int_member(req->params, "formid");
  dt_masks_form_t *form = dt_masks_get_from_id_ext(session->dev.forms, formid);
  if(!form)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Mask form not found");

  const dt_masks_type_t base_type = form->type & ~(DT_MASKS_CLONE | DT_MASKS_NON_CLONE);

  if(base_type == DT_MASKS_CIRCLE)
  {
    dt_masks_point_circle_t *circle = g_list_nth_data(form->points, 0);
    if(!circle)
      return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "Circle has no points");
    if(json_object_has_member(req->params, "center"))
    {
      JsonArray *c = json_object_get_array_member(req->params, "center");
      circle->center[0] = (float)json_array_get_double_element(c, 0);
      circle->center[1] = (float)json_array_get_double_element(c, 1);
    }
    if(json_object_has_member(req->params, "radius"))
      circle->radius = (float)json_object_get_double_member(req->params, "radius");
    if(json_object_has_member(req->params, "border"))
      circle->border = (float)json_object_get_double_member(req->params, "border");
  }
  else if(base_type == DT_MASKS_ELLIPSE)
  {
    dt_masks_point_ellipse_t *ellipse = g_list_nth_data(form->points, 0);
    if(!ellipse)
      return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "Ellipse has no points");
    if(json_object_has_member(req->params, "center"))
    {
      JsonArray *c = json_object_get_array_member(req->params, "center");
      ellipse->center[0] = (float)json_array_get_double_element(c, 0);
      ellipse->center[1] = (float)json_array_get_double_element(c, 1);
    }
    if(json_object_has_member(req->params, "radius"))
    {
      JsonArray *r = json_object_get_array_member(req->params, "radius");
      ellipse->radius[0] = (float)json_array_get_double_element(r, 0);
      ellipse->radius[1] = (float)json_array_get_double_element(r, 1);
    }
    if(json_object_has_member(req->params, "rotation"))
      ellipse->rotation = (float)json_object_get_double_member(req->params, "rotation");
    if(json_object_has_member(req->params, "border"))
      ellipse->border = (float)json_object_get_double_member(req->params, "border");
    if(json_object_has_member(req->params, "flags"))
      ellipse->flags = (dt_masks_ellipse_flags_t)json_object_get_int_member(req->params, "flags");
  }
  else if(base_type == DT_MASKS_GRADIENT)
  {
    dt_masks_point_gradient_t *gradient = g_list_nth_data(form->points, 0);
    if(!gradient)
      return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL, "Gradient has no points");
    if(json_object_has_member(req->params, "anchor"))
    {
      JsonArray *a = json_object_get_array_member(req->params, "anchor");
      gradient->anchor[0] = (float)json_array_get_double_element(a, 0);
      gradient->anchor[1] = (float)json_array_get_double_element(a, 1);
    }
    if(json_object_has_member(req->params, "rotation"))
      gradient->rotation = (float)json_object_get_double_member(req->params, "rotation");
    if(json_object_has_member(req->params, "compression"))
      gradient->compression = (float)json_object_get_double_member(req->params, "compression");
    if(json_object_has_member(req->params, "steepness"))
      gradient->steepness = (float)json_object_get_double_member(req->params, "steepness");
    if(json_object_has_member(req->params, "curvature"))
      gradient->curvature = (float)json_object_get_double_member(req->params, "curvature");
    if(json_object_has_member(req->params, "state"))
      gradient->state = (dt_masks_gradient_states_t)json_object_get_int_member(req->params, "state");
  }
  else
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "update_mask currently supports circle, ellipse, and gradient only");
  }

  // Opacity is stored on the group child entry, not on the form itself
  if(json_object_has_member(req->params, "opacity"))
  {
    const float opacity = (float)json_object_get_double_member(req->params, "opacity");
    // Find all group children referencing this form and update opacity
    for(GList *f = session->dev.forms; f; f = g_list_next(f))
    {
      dt_masks_form_t *grp = f->data;
      if((grp->type & DT_MASKS_GROUP) == 0) continue;
      for(GList *p = grp->points; p; p = g_list_next(p))
      {
        dt_masks_point_group_t *grpt = p->data;
        if(grpt->formid == formid)
          grpt->opacity = opacity;
      }
    }
  }

  const gboolean preview_only = json_object_has_member(req->params, "preview_only")
    && json_object_get_boolean_member(req->params, "preview_only");

  if(!preview_only)
  {
    dt_dev_add_masks_history_item_ext(&session->dev, NULL, TRUE, FALSE);
    dt_dev_write_history(&session->dev);
  }

  return dt_server_make_response(req->id, json_node_new(JSON_NODE_NULL));
}

char *dt_server_develop_assign_mask(dt_server_t *server, const dt_server_request_t *req)
{
  if(!req->params || !json_object_has_member(req->params, "session_id")
     || !json_object_has_member(req->params, "formid")
     || !json_object_has_member(req->params, "op"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing session_id, formid, or op parameter");

  const char *session_id = json_object_get_string_member(req->params, "session_id");
  if(!session_id)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "session_id must be a string");
  dt_server_session_t *session = dt_server_find_session(server, session_id);
  if(!session)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Session not found");

  const dt_mask_id_t formid = (dt_mask_id_t)json_object_get_int_member(req->params, "formid");
  const char *op = json_object_get_string_member(req->params, "op");
  const int instance = json_object_has_member(req->params, "instance")
    ? (int)json_object_get_int_member(req->params, "instance") : 0;

  dt_masks_form_t *form = dt_masks_get_from_id_ext(session->dev.forms, formid);
  if(!form)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Mask form not found");

  dt_iop_module_t *module = _find_module(session, op, instance);
  if(!module)
    return dt_server_make_error(req->id, DT_SERVER_ERR_NOT_FOUND, "Module not found");
  if(!(module->flags() & IOP_FLAGS_SUPPORTS_BLENDING) || !module->blend_params)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS, "Module does not support blending");

  // Get or create mask group
  dt_masks_form_t *grp = dt_masks_get_from_id_ext(session->dev.forms,
                                                   module->blend_params->mask_id);
  if(!grp || !(grp->type & DT_MASKS_GROUP))
  {
    grp = dt_masks_create(DT_MASKS_GROUP);
    _server_check_mask_id(&session->dev, grp);
    snprintf(grp->name, sizeof(grp->name), "group %s", op);
    session->dev.forms = g_list_append(session->dev.forms, grp);
    module->blend_params->mask_id = grp->formid;
  }

  // Check if already in group
  for(GList *p = grp->points; p; p = g_list_next(p))
  {
    dt_masks_point_group_t *grpt = p->data;
    if(grpt->formid == formid)
      return dt_server_make_response(req->id, json_node_new(JSON_NODE_NULL));
  }

  // Add form to group
  dt_masks_point_group_t *grpt = malloc(sizeof(dt_masks_point_group_t));
  grpt->formid = form->formid;
  grpt->parentid = grp->formid;
  grpt->state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE;
  if(grp->points)
    grpt->state |= DT_MASKS_STATE_UNION;
  grpt->opacity = json_object_has_member(req->params, "opacity")
    ? (float)json_object_get_double_member(req->params, "opacity") : 1.0f;
  grp->points = g_list_append(grp->points, grpt);

  // Enable drawn mask mode
  if(!(module->blend_params->mask_mode & DEVELOP_MASK_MASK))
    module->blend_params->mask_mode |= DEVELOP_MASK_MASK;

  dt_iop_commit_blend_params(module, module->blend_params);
  dt_dev_add_history_item_ext(&session->dev, module, module->enabled, TRUE);
  dt_dev_add_masks_history_item_ext(&session->dev, NULL, TRUE, FALSE);
  dt_dev_write_history(&session->dev);

  return dt_server_make_response(req->id, json_node_new(JSON_NODE_NULL));
}
