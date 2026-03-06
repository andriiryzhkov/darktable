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
#include "common/iop_order.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/pixelpipe.h"

#include <float.h>
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

// Mirror of dt_iop_sigmoid_params_t from iop/sigmoid.c
// Must match the struct layout exactly (introspection version 3).
typedef enum _server_sigmoid_method_t
{
  _SIGMOID_METHOD_PER_CHANNEL = 0,
  _SIGMOID_METHOD_RGB_RATIO = 1
} _server_sigmoid_method_t;

typedef struct _server_sigmoid_params_t
{
  float middle_grey_contrast;
  float contrast_skewness;
  float display_white_target;
  float display_black_target;
  _server_sigmoid_method_t color_processing;
  float hue_preservation;
  float red_inset;
  float red_rotation;
  float green_inset;
  float green_rotation;
  float blue_inset;
  float blue_rotation;
  float purity;
  int base_primaries;
} _server_sigmoid_params_t;

// Mirror of dt_iop_rawprepare_params_t from iop/rawprepare.c
// Must match the struct layout exactly (introspection version 2).
typedef enum _server_rawprepare_flat_field_t
{
  _FLAT_FIELD_OFF = 0,
  _FLAT_FIELD_EMBEDDED = 1
} _server_rawprepare_flat_field_t;

typedef struct _server_rawprepare_params_t
{
  int32_t left;
  int32_t top;
  int32_t right;
  int32_t bottom;
  uint16_t raw_black_level_separate[4];
  uint16_t raw_white_point;
  _server_rawprepare_flat_field_t flat_field;
} _server_rawprepare_params_t;

// Mirror of dt_iop_colorin_params_t from iop/colorin.c
// Must match the struct layout exactly (introspection version 7).
#define _SERVER_IOP_COLOR_ICC_LEN 512

typedef enum _server_color_normalize_t
{
  _NORMALIZE_OFF = 0,
  _NORMALIZE_SRGB = 1,
  _NORMALIZE_ADOBE_RGB = 2,
  _NORMALIZE_LINEAR_REC709_RGB = 3,
  _NORMALIZE_LINEAR_REC2020_RGB = 4
} _server_color_normalize_t;

typedef struct _server_colorin_params_t
{
  dt_colorspaces_color_profile_type_t type;
  char filename[_SERVER_IOP_COLOR_ICC_LEN];
  dt_iop_color_intent_t intent;
  _server_color_normalize_t normalize;
  gboolean blue_mapping;
  dt_colorspaces_color_profile_type_t type_work;
  char filename_work[_SERVER_IOP_COLOR_ICC_LEN];
} _server_colorin_params_t;

// Mirror of dt_iop_colorout_params_t from iop/colorout.c
// Must match the struct layout exactly (introspection version 5).
typedef struct _server_colorout_params_t
{
  dt_colorspaces_color_profile_type_t type;
  char filename[_SERVER_IOP_COLOR_ICC_LEN];
  dt_iop_color_intent_t intent;
} _server_colorout_params_t;

// Mirror of dt_iop_demosaic_params_t from iop/demosaic.c
// Must match the struct layout exactly (introspection version 5).
#define _SERVER_DEMOSAIC_XTRANS 1024
#define _SERVER_DEMOSAIC_DUAL   2048

typedef enum _server_demosaic_greeneq_t
{
  _DEMOSAIC_GREEN_EQ_NO = 0,
  _DEMOSAIC_GREEN_EQ_LOCAL = 1,
  _DEMOSAIC_GREEN_EQ_FULL = 2,
  _DEMOSAIC_GREEN_EQ_BOTH = 3
} _server_demosaic_greeneq_t;

typedef enum _server_demosaic_smooth_t
{
  _DEMOSAIC_SMOOTH_OFF = 0,
  _DEMOSAIC_SMOOTH_1 = 1,
  _DEMOSAIC_SMOOTH_2 = 2,
  _DEMOSAIC_SMOOTH_3 = 3,
  _DEMOSAIC_SMOOTH_4 = 4,
  _DEMOSAIC_SMOOTH_5 = 5
} _server_demosaic_smooth_t;

typedef enum _server_demosaic_method_t
{
  _DEMOSAIC_PPG = 0,
  _DEMOSAIC_AMAZE = 1,
  _DEMOSAIC_VNG4 = 2,
  _DEMOSAIC_PASSTHROUGH_MONOCHROME = 3,
  _DEMOSAIC_PASSTHROUGH_COLOR = 4,
  _DEMOSAIC_RCD = 5,
  _DEMOSAIC_LMMSE = 6,
  _DEMOSAIC_MONO = 7,
  _DEMOSAIC_RCD_DUAL = _SERVER_DEMOSAIC_DUAL | 5,
  _DEMOSAIC_AMAZE_DUAL = _SERVER_DEMOSAIC_DUAL | 1,
  _DEMOSAIC_VNG = _SERVER_DEMOSAIC_XTRANS | 0,
  _DEMOSAIC_MARKESTEIJN = _SERVER_DEMOSAIC_XTRANS | 1,
  _DEMOSAIC_MARKESTEIJN_3 = _SERVER_DEMOSAIC_XTRANS | 2,
  _DEMOSAIC_PASSTHR_MONOX = _SERVER_DEMOSAIC_XTRANS | 3,
  _DEMOSAIC_FDC = _SERVER_DEMOSAIC_XTRANS | 4,
  _DEMOSAIC_PASSTHR_COLORX = _SERVER_DEMOSAIC_XTRANS | 5,
  _DEMOSAIC_MARKEST3_DUAL = _SERVER_DEMOSAIC_DUAL | _SERVER_DEMOSAIC_XTRANS | 2
} _server_demosaic_method_t;

typedef enum _server_demosaic_lmmse_t
{
  _DEMOSAIC_LMMSE_REFINE_0 = 0,
  _DEMOSAIC_LMMSE_REFINE_1 = 1,
  _DEMOSAIC_LMMSE_REFINE_2 = 2,
  _DEMOSAIC_LMMSE_REFINE_3 = 3,
  _DEMOSAIC_LMMSE_REFINE_4 = 4
} _server_demosaic_lmmse_t;

typedef struct _server_demosaic_params_t
{
  _server_demosaic_greeneq_t green_eq;
  float median_thrs;
  _server_demosaic_smooth_t color_smoothing;
  _server_demosaic_method_t demosaicing_method;
  _server_demosaic_lmmse_t lmmse_refine;
  float dual_thrs;
  float cs_radius;
  float cs_thrs;
  float cs_boost;
  int cs_iter;
  float cs_center;
  gboolean cs_enabled;
} _server_demosaic_params_t;

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

  // Clamp preview dimensions
  if(preview_width < 320) preview_width = 320;
  if(preview_width > 4096) preview_width = 4096;
  if(preview_height < 240) preview_height = 240;
  if(preview_height > 4096) preview_height = 4096;

  // Single-client model: close all existing sessions before opening a new one.
  // This prevents orphaned sessions from accumulating when the client rapidly
  // switches images (the previous close may not have arrived yet).
  for(int i = server->session_count - 1; i >= 0; i--)
  {
    if(server->sessions[i])
    {
      dt_server_session_t *old = server->sessions[i];
      fprintf(stderr, "[server] develop.open: auto-closing session %s\n", old->session_id);
      dt_shm_destroy(&old->shm_buffers[0]);
      dt_shm_destroy(&old->shm_buffers[1]);
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

  // Create double-buffered SHM for preview frames
  char shm_name0[32], shm_name1[32];
  snprintf(shm_name0, sizeof(shm_name0), "/dt-prev-%s-0", session->session_id);
  snprintf(shm_name1, sizeof(shm_name1), "/dt-prev-%s-1", session->session_id);

  if(!dt_shm_create(&session->shm_buffers[0], shm_name0, preview_width, preview_height)
     || !dt_shm_create(&session->shm_buffers[1], shm_name1, preview_width, preview_height))
  {
    dt_shm_destroy(&session->shm_buffers[0]);
    dt_shm_destroy(&session->shm_buffers[1]);
    dt_dev_cleanup(&session->dev);
    // Remove session from slot
    server->sessions[server->session_count - 1] = NULL;
    server->session_count--;
    g_free(session);
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "Failed to create shared memory buffers");
  }

  session->front_buffer = 0;

  fprintf(stderr, "[server] develop.open: session=%s imgid=%d preview=%dx%d\n",
          session->session_id, imgid, preview_width, preview_height);

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
  json_builder_add_string_value(b, shm_name0);
  json_builder_add_string_value(b, shm_name1);
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

  fprintf(stderr, "[server] develop.close: session=%s\n", session_id);

  // Destroy SHM buffers
  dt_shm_destroy(&session->shm_buffers[0]);
  dt_shm_destroy(&session->shm_buffers[1]);

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
    const _server_exposure_params_t *p = (const _server_exposure_params_t *)target->params;
    json_builder_set_member_name(b, "mode");
    json_builder_add_int_value(b, (int)p->mode);
    json_builder_set_member_name(b, "exposure");
    json_builder_add_double_value(b, p->exposure);
    json_builder_set_member_name(b, "black");
    json_builder_add_double_value(b, p->black);
    json_builder_set_member_name(b, "compensate_exposure_bias");
    json_builder_add_boolean_value(b, p->compensate_exposure_bias);
    json_builder_set_member_name(b, "compensate_hilite_pres");
    json_builder_add_boolean_value(b, p->compensate_hilite_pres);
    json_builder_set_member_name(b, "deflicker_percentile");
    json_builder_add_double_value(b, p->deflicker_percentile);
    json_builder_set_member_name(b, "deflicker_target_level");
    json_builder_add_double_value(b, p->deflicker_target_level);

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
    if(p->mode == _EXPOSURE_MODE_DEFLICKER && !dt_pthread_mutex_trylock(&pipe->busy_mutex))
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
    const _server_temperature_params_t *p = (const _server_temperature_params_t *)target->params;
    json_builder_set_member_name(b, "red");
    json_builder_add_double_value(b, p->red);
    json_builder_set_member_name(b, "green");
    json_builder_add_double_value(b, p->green);
    json_builder_set_member_name(b, "blue");
    json_builder_add_double_value(b, p->blue);
    json_builder_set_member_name(b, "various");
    json_builder_add_double_value(b, p->various);
    json_builder_set_member_name(b, "preset");
    json_builder_add_int_value(b, p->preset);

    // Compute temperature and tint from coefficients using camera color matrix
    // Uses exact same method as darktable's _mul2temp (coeffs → XYZ → binary search)
    {
      float d65_cm[9];
      memcpy(d65_cm, session->dev.image_storage.d65_color_matrix, sizeof(d65_cm));
      double CAM_to_XYZ[3][4], XYZ_to_CAM[4][3];
      if(dt_colorspaces_conversion_matrices_xyz(
          session->dev.image_storage.adobe_XYZ_to_CAM, d65_cm,
          XYZ_to_CAM, CAM_to_XYZ))
      {
        // Invert coefficients: camera multipliers → XYZ (same as _mul2xyz)
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
  else if(!strcmp(op, "rawprepare"))
  {
    const _server_rawprepare_params_t *p = (const _server_rawprepare_params_t *)target->params;
    json_builder_set_member_name(b, "raw_black_level_separate");
    json_builder_begin_array(b);
    for(int i = 0; i < 4; i++)
      json_builder_add_int_value(b, p->raw_black_level_separate[i]);
    json_builder_end_array(b);
    json_builder_set_member_name(b, "raw_white_point");
    json_builder_add_int_value(b, p->raw_white_point);
    json_builder_set_member_name(b, "flat_field");
    json_builder_add_int_value(b, (int)p->flat_field);
    json_builder_set_member_name(b, "left");
    json_builder_add_int_value(b, p->left);
    json_builder_set_member_name(b, "top");
    json_builder_add_int_value(b, p->top);
    json_builder_set_member_name(b, "right");
    json_builder_add_int_value(b, p->right);
    json_builder_set_member_name(b, "bottom");
    json_builder_add_int_value(b, p->bottom);
  }
  else if(!strcmp(op, "colorin"))
  {
    const _server_colorin_params_t *p = (const _server_colorin_params_t *)target->params;

    // Current params
    json_builder_set_member_name(b, "type");
    json_builder_add_int_value(b, (int)p->type);
    json_builder_set_member_name(b, "filename");
    json_builder_add_string_value(b, p->filename);
    json_builder_set_member_name(b, "intent");
    json_builder_add_int_value(b, (int)p->intent);
    json_builder_set_member_name(b, "normalize");
    json_builder_add_int_value(b, (int)p->normalize);
    json_builder_set_member_name(b, "type_work");
    json_builder_add_int_value(b, (int)p->type_work);
    json_builder_set_member_name(b, "filename_work");
    json_builder_add_string_value(b, p->filename_work);

    // Current profile display names
    json_builder_set_member_name(b, "input_profile_name");
    json_builder_add_string_value(b, dt_colorspaces_get_name(p->type, p->filename));
    json_builder_set_member_name(b, "work_profile_name");
    json_builder_add_string_value(b, dt_colorspaces_get_name(p->type_work, p->filename_work));

    // Build available input profiles list (system profiles with in_pos)
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

    // Build available working profiles list
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
    const _server_colorout_params_t *p = (const _server_colorout_params_t *)target->params;

    json_builder_set_member_name(b, "type");
    json_builder_add_int_value(b, (int)p->type);
    json_builder_set_member_name(b, "filename");
    json_builder_add_string_value(b, p->filename);
    json_builder_set_member_name(b, "intent");
    json_builder_add_int_value(b, (int)p->intent);

    // Current profile display name
    json_builder_set_member_name(b, "output_profile_name");
    json_builder_add_string_value(b, dt_colorspaces_get_name(p->type, p->filename));

    // Available output profiles
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
    const _server_demosaic_params_t *p = (const _server_demosaic_params_t *)target->params;
    json_builder_set_member_name(b, "demosaicing_method");
    json_builder_add_int_value(b, (int)p->demosaicing_method);
    json_builder_set_member_name(b, "green_eq");
    json_builder_add_int_value(b, (int)p->green_eq);
    json_builder_set_member_name(b, "median_thrs");
    json_builder_add_double_value(b, p->median_thrs);
    json_builder_set_member_name(b, "color_smoothing");
    json_builder_add_int_value(b, (int)p->color_smoothing);
    json_builder_set_member_name(b, "lmmse_refine");
    json_builder_add_int_value(b, (int)p->lmmse_refine);
    json_builder_set_member_name(b, "dual_thrs");
    json_builder_add_double_value(b, p->dual_thrs);
    json_builder_set_member_name(b, "cs_enabled");
    json_builder_add_boolean_value(b, p->cs_enabled);
    json_builder_set_member_name(b, "cs_radius");
    json_builder_add_double_value(b, p->cs_radius);
    json_builder_set_member_name(b, "cs_thrs");
    json_builder_add_double_value(b, p->cs_thrs);
    json_builder_set_member_name(b, "cs_boost");
    json_builder_add_double_value(b, p->cs_boost);
    json_builder_set_member_name(b, "cs_iter");
    json_builder_add_int_value(b, p->cs_iter);
    json_builder_set_member_name(b, "cs_center");
    json_builder_add_double_value(b, p->cs_center);

    // Include sensor type so UI can show appropriate method options
    const dt_image_t *img = &session->dev.image_storage;
    const gboolean is_xtrans = img->buf_dsc.filters == 9u;
    const gboolean is_bayer4 = img->flags & DT_IMAGE_4BAYER;
    const gboolean is_mono = dt_image_is_monochrome(img);
    const char *sensor_type = is_mono ? "mono" : is_xtrans ? "xtrans" : is_bayer4 ? "bayer4" : "bayer";
    json_builder_set_member_name(b, "sensor_type");
    json_builder_add_string_value(b, sensor_type);
  }
  else if(!strcmp(op, "sigmoid"))
  {
    const _server_sigmoid_params_t *p = (const _server_sigmoid_params_t *)target->params;
    json_builder_set_member_name(b, "middle_grey_contrast");
    json_builder_add_double_value(b, p->middle_grey_contrast);
    json_builder_set_member_name(b, "contrast_skewness");
    json_builder_add_double_value(b, p->contrast_skewness);
    json_builder_set_member_name(b, "color_processing");
    json_builder_add_int_value(b, (int)p->color_processing);
    json_builder_set_member_name(b, "hue_preservation");
    json_builder_add_double_value(b, p->hue_preservation);
    json_builder_set_member_name(b, "display_white_target");
    json_builder_add_double_value(b, p->display_white_target);
    json_builder_set_member_name(b, "display_black_target");
    json_builder_add_double_value(b, p->display_black_target);
    json_builder_set_member_name(b, "base_primaries");
    json_builder_add_int_value(b, p->base_primaries);
    json_builder_set_member_name(b, "red_inset");
    json_builder_add_double_value(b, p->red_inset);
    json_builder_set_member_name(b, "red_rotation");
    json_builder_add_double_value(b, p->red_rotation);
    json_builder_set_member_name(b, "green_inset");
    json_builder_add_double_value(b, p->green_inset);
    json_builder_set_member_name(b, "green_rotation");
    json_builder_add_double_value(b, p->green_rotation);
    json_builder_set_member_name(b, "blue_inset");
    json_builder_add_double_value(b, p->blue_inset);
    json_builder_set_member_name(b, "blue_rotation");
    json_builder_add_double_value(b, p->blue_rotation);
    json_builder_set_member_name(b, "purity");
    json_builder_add_double_value(b, p->purity);
  }
  else
  {
    // Generic: return params as base64 blob for unsupported modules
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

    // Pipeline is current — write to SHM and send event
    dt_pthread_mutex_lock(&pipe->backbuf_mutex);

    if(!pipe->backbuf || pipe->backbuf_width <= 0 || pipe->backbuf_height <= 0)
    {
      dt_pthread_mutex_unlock(&pipe->backbuf_mutex);
      fprintf(stderr, "[server] async pipeline: no output\n");
      break;
    }

    const int rendered_width = pipe->backbuf_width;
    const int rendered_height = pipe->backbuf_height;

    const int back = 1 - session->front_buffer;
    dt_shm_buffer_t *shm = &session->shm_buffers[back];

    session->frame_sequence++;

    dt_shm_write_header(shm, rendered_width, rendered_height,
                         DT_SHM_FORMAT_BGRA8, session->frame_sequence);

    uint8_t *dst = dt_shm_pixel_data(shm);
    const size_t copy_size = (size_t)rendered_width * rendered_height * 4;
    memcpy(dst, pipe->backbuf, copy_size);

    dt_pthread_mutex_unlock(&pipe->backbuf_mutex);

    __atomic_store_n(&shm->mapped->ready, 1, __ATOMIC_RELEASE);
    session->front_buffer = back;
    session->dirty = FALSE;

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
    // Worker is already running — it will see the bumped pipeline_seq and reprocess
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
  else if(!strcmp(op, "exposure"))
  {
    _server_exposure_params_t *p = (_server_exposure_params_t *)target->params;

    if(json_object_has_member(new_params, "mode"))
      p->mode = (_server_exposure_mode_t)json_object_get_int_member(new_params, "mode");
    if(json_object_has_member(new_params, "exposure"))
      p->exposure = (float)json_object_get_double_member(new_params, "exposure");
    if(json_object_has_member(new_params, "black"))
      p->black = (float)json_object_get_double_member(new_params, "black");
    if(json_object_has_member(new_params, "compensate_exposure_bias"))
      p->compensate_exposure_bias = json_object_get_boolean_member(new_params, "compensate_exposure_bias");
    if(json_object_has_member(new_params, "compensate_hilite_pres"))
      p->compensate_hilite_pres = json_object_get_boolean_member(new_params, "compensate_hilite_pres");
    if(json_object_has_member(new_params, "deflicker_percentile"))
      p->deflicker_percentile = (float)json_object_get_double_member(new_params, "deflicker_percentile");
    if(json_object_has_member(new_params, "deflicker_target_level"))
      p->deflicker_target_level = (float)json_object_get_double_member(new_params, "deflicker_target_level");
  }
  else if(!strcmp(op, "temperature"))
  {
    _server_temperature_params_t *p = (_server_temperature_params_t *)target->params;

    if(json_object_has_member(new_params, "red"))
      p->red = (float)json_object_get_double_member(new_params, "red");
    if(json_object_has_member(new_params, "green"))
      p->green = (float)json_object_get_double_member(new_params, "green");
    if(json_object_has_member(new_params, "blue"))
      p->blue = (float)json_object_get_double_member(new_params, "blue");
    if(json_object_has_member(new_params, "various"))
      p->various = (float)json_object_get_double_member(new_params, "various");
    if(json_object_has_member(new_params, "preset"))
    {
      const int preset = (int)json_object_get_int_member(new_params, "preset");
      p->preset = preset;

      // When preset changes, load corresponding coefficients from dev->chroma
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
      // Recover current temp/tint from coefficients using binary search (same as get_params)
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
        // Only update 4th channel if sensor uses it (non-zero original value)
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
  else if(!strcmp(op, "rawprepare"))
  {
    _server_rawprepare_params_t *p = (_server_rawprepare_params_t *)target->params;

    if(json_object_has_member(new_params, "raw_black_level_separate"))
    {
      JsonArray *arr = json_object_get_array_member(new_params, "raw_black_level_separate");
      for(int i = 0; i < 4 && i < (int)json_array_get_length(arr); i++)
        p->raw_black_level_separate[i] = (uint16_t)json_array_get_int_element(arr, i);
    }
    if(json_object_has_member(new_params, "raw_white_point"))
      p->raw_white_point = (uint16_t)json_object_get_int_member(new_params, "raw_white_point");
    if(json_object_has_member(new_params, "flat_field"))
      p->flat_field = (_server_rawprepare_flat_field_t)json_object_get_int_member(new_params, "flat_field");
    if(json_object_has_member(new_params, "left"))
      p->left = (int32_t)json_object_get_int_member(new_params, "left");
    if(json_object_has_member(new_params, "top"))
      p->top = (int32_t)json_object_get_int_member(new_params, "top");
    if(json_object_has_member(new_params, "right"))
      p->right = (int32_t)json_object_get_int_member(new_params, "right");
    if(json_object_has_member(new_params, "bottom"))
      p->bottom = (int32_t)json_object_get_int_member(new_params, "bottom");
  }
  else if(!strcmp(op, "colorin"))
  {
    _server_colorin_params_t *p = (_server_colorin_params_t *)target->params;

    if(json_object_has_member(new_params, "type"))
      p->type = (dt_colorspaces_color_profile_type_t)json_object_get_int_member(new_params, "type");
    if(json_object_has_member(new_params, "filename"))
      g_strlcpy(p->filename, json_object_get_string_member(new_params, "filename"), _SERVER_IOP_COLOR_ICC_LEN);
    if(json_object_has_member(new_params, "intent"))
      p->intent = (dt_iop_color_intent_t)json_object_get_int_member(new_params, "intent");
    if(json_object_has_member(new_params, "normalize"))
      p->normalize = (_server_color_normalize_t)json_object_get_int_member(new_params, "normalize");
    if(json_object_has_member(new_params, "type_work"))
      p->type_work = (dt_colorspaces_color_profile_type_t)json_object_get_int_member(new_params, "type_work");
    if(json_object_has_member(new_params, "filename_work"))
      g_strlcpy(p->filename_work, json_object_get_string_member(new_params, "filename_work"), _SERVER_IOP_COLOR_ICC_LEN);
  }
  else if(!strcmp(op, "colorout"))
  {
    _server_colorout_params_t *p = (_server_colorout_params_t *)target->params;

    if(json_object_has_member(new_params, "type"))
      p->type = (dt_colorspaces_color_profile_type_t)json_object_get_int_member(new_params, "type");
    if(json_object_has_member(new_params, "filename"))
      g_strlcpy(p->filename, json_object_get_string_member(new_params, "filename"), _SERVER_IOP_COLOR_ICC_LEN);
    if(json_object_has_member(new_params, "intent"))
      p->intent = (dt_iop_color_intent_t)json_object_get_int_member(new_params, "intent");
  }
  else if(!strcmp(op, "demosaic"))
  {
    _server_demosaic_params_t *p = (_server_demosaic_params_t *)target->params;

    if(json_object_has_member(new_params, "demosaicing_method"))
      p->demosaicing_method = (_server_demosaic_method_t)json_object_get_int_member(new_params, "demosaicing_method");
    if(json_object_has_member(new_params, "green_eq"))
      p->green_eq = (_server_demosaic_greeneq_t)json_object_get_int_member(new_params, "green_eq");
    if(json_object_has_member(new_params, "median_thrs"))
      p->median_thrs = (float)json_object_get_double_member(new_params, "median_thrs");
    if(json_object_has_member(new_params, "color_smoothing"))
      p->color_smoothing = (_server_demosaic_smooth_t)json_object_get_int_member(new_params, "color_smoothing");
    if(json_object_has_member(new_params, "lmmse_refine"))
      p->lmmse_refine = (_server_demosaic_lmmse_t)json_object_get_int_member(new_params, "lmmse_refine");
    if(json_object_has_member(new_params, "dual_thrs"))
      p->dual_thrs = (float)json_object_get_double_member(new_params, "dual_thrs");
    if(json_object_has_member(new_params, "cs_enabled"))
      p->cs_enabled = json_object_get_boolean_member(new_params, "cs_enabled");
    if(json_object_has_member(new_params, "cs_radius"))
      p->cs_radius = (float)json_object_get_double_member(new_params, "cs_radius");
    if(json_object_has_member(new_params, "cs_thrs"))
      p->cs_thrs = (float)json_object_get_double_member(new_params, "cs_thrs");
    if(json_object_has_member(new_params, "cs_boost"))
      p->cs_boost = (float)json_object_get_double_member(new_params, "cs_boost");
    if(json_object_has_member(new_params, "cs_iter"))
      p->cs_iter = (int)json_object_get_int_member(new_params, "cs_iter");
    if(json_object_has_member(new_params, "cs_center"))
      p->cs_center = (float)json_object_get_double_member(new_params, "cs_center");
  }
  else if(!strcmp(op, "sigmoid"))
  {
    _server_sigmoid_params_t *p = (_server_sigmoid_params_t *)target->params;

    if(json_object_has_member(new_params, "middle_grey_contrast"))
      p->middle_grey_contrast = (float)json_object_get_double_member(new_params, "middle_grey_contrast");
    if(json_object_has_member(new_params, "contrast_skewness"))
      p->contrast_skewness = (float)json_object_get_double_member(new_params, "contrast_skewness");
    if(json_object_has_member(new_params, "color_processing"))
      p->color_processing = (_server_sigmoid_method_t)json_object_get_int_member(new_params, "color_processing");
    if(json_object_has_member(new_params, "hue_preservation"))
      p->hue_preservation = (float)json_object_get_double_member(new_params, "hue_preservation");
    if(json_object_has_member(new_params, "display_white_target"))
      p->display_white_target = (float)json_object_get_double_member(new_params, "display_white_target");
    if(json_object_has_member(new_params, "display_black_target"))
      p->display_black_target = (float)json_object_get_double_member(new_params, "display_black_target");
    if(json_object_has_member(new_params, "base_primaries"))
      p->base_primaries = (int)json_object_get_int_member(new_params, "base_primaries");
    if(json_object_has_member(new_params, "red_inset"))
      p->red_inset = (float)json_object_get_double_member(new_params, "red_inset");
    if(json_object_has_member(new_params, "red_rotation"))
      p->red_rotation = (float)json_object_get_double_member(new_params, "red_rotation");
    if(json_object_has_member(new_params, "green_inset"))
      p->green_inset = (float)json_object_get_double_member(new_params, "green_inset");
    if(json_object_has_member(new_params, "green_rotation"))
      p->green_rotation = (float)json_object_get_double_member(new_params, "green_rotation");
    if(json_object_has_member(new_params, "blue_inset"))
      p->blue_inset = (float)json_object_get_double_member(new_params, "blue_inset");
    if(json_object_has_member(new_params, "blue_rotation"))
      p->blue_rotation = (float)json_object_get_double_member(new_params, "blue_rotation");
    if(json_object_has_member(new_params, "purity"))
      p->purity = (float)json_object_get_double_member(new_params, "purity");
  }
  else
  {
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Module params not yet supported for this operation");
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

  // Process the pipeline (synchronous — blocks until done)
  dt_dev_pixelpipe_t *pipe = session->dev.full.pipe;

  fprintf(stderr, "[server] develop.request_preview: session=%s processing...\n",
          session_id);

  dt_dev_process_image_job(&session->dev, &session->dev.full, pipe, -1, DT_DEVICE_CPU);

  // Lock backbuf_mutex to safely access pipeline output
  dt_pthread_mutex_lock(&pipe->backbuf_mutex);

  if(!pipe->backbuf || pipe->backbuf_width <= 0 || pipe->backbuf_height <= 0)
  {
    dt_pthread_mutex_unlock(&pipe->backbuf_mutex);
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "Pipeline processing failed — no output");
  }

  const int rendered_width = pipe->backbuf_width;
  const int rendered_height = pipe->backbuf_height;

  fprintf(stderr, "[server] develop.request_preview: rendered %dx%d\n",
          rendered_width, rendered_height);

  // Write to back buffer (the one client is NOT reading)
  const int back = 1 - session->front_buffer;
  dt_shm_buffer_t *shm = &session->shm_buffers[back];

  session->frame_sequence++;

  // Update SHM header
  dt_shm_write_header(shm, rendered_width, rendered_height,
                       DT_SHM_FORMAT_BGRA8, session->frame_sequence);

  // Copy pixel data while holding the lock
  uint8_t *dst = dt_shm_pixel_data(shm);
  const size_t copy_size = (size_t)rendered_width * rendered_height * 4;
  memcpy(dst, pipe->backbuf, copy_size);

  dt_pthread_mutex_unlock(&pipe->backbuf_mutex);

  // Mark buffer as ready and swap
  __atomic_store_n(&shm->mapped->ready, 1, __ATOMIC_RELEASE);
  session->front_buffer = back;
  session->dirty = FALSE;

  // Queue a preview_ready event for the client
  {
    JsonBuilder *eb = json_builder_new();
    json_builder_begin_object(eb);

    json_builder_set_member_name(eb, "session_id");
    json_builder_add_string_value(eb, session->session_id);

    json_builder_set_member_name(eb, "shm_name");
    json_builder_add_string_value(eb, shm->name);

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

  // Build the direct response (client also gets the event asynchronously)
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "session_id");
  json_builder_add_string_value(b, session->session_id);

  json_builder_set_member_name(b, "width");
  json_builder_add_int_value(b, rendered_width);

  json_builder_set_member_name(b, "height");
  json_builder_add_int_value(b, rendered_height);

  json_builder_set_member_name(b, "sequence");
  json_builder_add_int_value(b, session->frame_sequence);

  json_builder_set_member_name(b, "shm_name");
  json_builder_add_string_value(b, shm->name);

  json_builder_set_member_name(b, "front_buffer");
  json_builder_add_int_value(b, session->front_buffer);

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
