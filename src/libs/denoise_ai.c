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

#include "ai/backend.h"
#include "common/ai_models.h"
#include "bauhaus/bauhaus.h"
#include "control/jobs/control_jobs.h"
#include "control/signal.h"
#include "common/collection.h"
#include "common/film.h"

#include "develop/develop.h"
#include "imageio/imageio_module.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <tiffio.h>

DT_MODULE(1)

// Config key for the model ID to use
#define CONF_MODEL_KEY "plugins/lighttable/denoise_ai/model"
#define DEFAULT_MODEL_ID "nafnet-sidd-width32"

/**
 * @struct dt_lib_denoise_ai_t
 * @brief GUI Data for the Denoise AI module.
 */
typedef struct dt_lib_denoise_ai_t
{
  GtkBox *box;
  GtkWidget *sigma_slider;
  GtkWidget *button;
  dt_ai_environment_t *env;
  char *model_id;           // Model ID from config (owned)
  gboolean model_available; // TRUE if the configured model was found
  gboolean job_running;     // TRUE while a denoise job is in progress
} dt_lib_denoise_ai_t;

/**
 * @struct dt_denoise_job_t
 * @brief Job Data passed to the background worker.
 */
typedef struct dt_denoise_job_t
{
  char *model_id;
  dt_ai_environment_t *env;
  GList *images;
  dt_job_t *control_job;
  dt_ai_context_t *ctx;
  float sigma;
  dt_ai_provider_t provider;
  dt_lib_module_t *self; // back-pointer for UI update on completion
} dt_denoise_job_t;

typedef struct dt_denoise_format_params_t
{
  dt_imageio_module_data_t parent;
  dt_denoise_job_t *job;
} dt_denoise_format_params_t;

const char *name(dt_lib_module_t *self) { return _("ai denoise"); }
const char *description(dt_lib_module_t *self)
{
  return _(
    "denoise the current image\n"
    "using generative ai models");
}
dt_view_type_flags_t views(dt_lib_module_t *self) { return DT_VIEW_DARKROOM; }
uint32_t container(dt_lib_module_t *self) { return DT_UI_CONTAINER_PANEL_LEFT_CENTER; }
int position(const dt_lib_module_t *self) { return 1000; }

static int _ai_check_bpp(dt_imageio_module_data_t *data) { return 32; }
static int _ai_check_levels(dt_imageio_module_data_t *data)
{
  return IMAGEIO_RGB | IMAGEIO_FLOAT;
}
static const char *_ai_get_mime(dt_imageio_module_data_t *data) { return "memory"; }

// sRGB transfer functions for model input/output conversion
static inline float _linear_to_srgb(float v)
{
  if(v <= 0.0f)
    return 0.0f;
  if(v >= 1.0f)
    return 1.0f;
  return (v <= 0.0031308f) ? 12.92f * v : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
}

static inline float _srgb_to_linear(float v)
{
  if(v <= 0.0f)
    return 0.0f;
  if(v >= 1.0f)
    return 1.0f;
  return (v <= 0.04045f) ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
}

static int _run_patch(
  dt_ai_context_t *ctx,
  float *in_patch,
  int w,
  int h,
  float *out_patch,
  int tile_idx,
  float sigma)
{

  const int total_pixels = w * h * 3;
  const int num_inputs = dt_ai_get_input_count(ctx);

  // Clip and convert linear -> sRGB (model expects sRGB)
  for(int i = 0; i < total_pixels; i++)
  {
    float v = in_patch[i];
    if(v < 0.0f)
      v = 0.0f;
    if(v > 1.0f)
      v = 1.0f;
    in_patch[i] = _linear_to_srgb(v);
  }

  // Brightness-boost dark tiles so the model sees values in its trained range.
  // NAFNet (and similar models) diverge on uniformly dark tiles (max sRGB < ~0.3).
  // We scale values up before inference and scale the output back down.
  const float DARK_THRESHOLD = 0.3f;
  const float BOOST_TARGET = 0.5f;
  const float MAX_BOOST = 3.0f;
  float srgb_max = 0.0f;
  for(int i = 0; i < total_pixels; i++)
  {
    if(in_patch[i] > srgb_max)
      srgb_max = in_patch[i];
  }

  float boost = 1.0f;
  if(srgb_max > 0.0f && srgb_max < DARK_THRESHOLD)
  {
    boost = BOOST_TARGET / srgb_max;
    if(boost > MAX_BOOST)
      boost = MAX_BOOST;
    for(int i = 0; i < total_pixels; i++)
      in_patch[i] *= boost;
    dt_print(
      DT_DEBUG_AI,
      "[denoise_ai] Tile %d: dark tile (max sRGB=%.4f), boost=%.2fx%s, boosted max=%.4f",
      tile_idx,
      srgb_max,
      boost,
      (BOOST_TARGET / srgb_max > MAX_BOOST) ? " (capped)" : "",
      srgb_max * boost);
  }

// Image input: BCHW {1, 3, H, W}
// Max 4 inputs should cover all known model architectures
#define MAX_MODEL_INPUTS 4
  if(num_inputs > MAX_MODEL_INPUTS)
    return 1;
  int64_t input_shape[] = {1, 3, h, w};
  dt_ai_tensor_t inputs[MAX_MODEL_INPUTS];
  memset(inputs, 0, sizeof(inputs));
  inputs[0] = (dt_ai_tensor_t){.data = (void *)in_patch,
                               .shape = input_shape,
                               .ndim = 4,
                               .type = DT_AI_FLOAT};

  // Noise level map for models that need it (e.g. FFDNet): {1, 1, H, W}
  float *noise_map = NULL;
  int64_t noise_shape[] = {1, 1, h, w};
  if(num_inputs >= 2)
  {
    const size_t map_size = (size_t)w * h;
    noise_map = g_try_malloc(map_size * sizeof(float));
    if(!noise_map)
      return 1;
    const float sigma_norm = sigma / 255.0f;
    for(size_t i = 0; i < map_size; i++)
      noise_map[i] = sigma_norm;
    inputs[1] = (dt_ai_tensor_t){.data = (void *)noise_map,
                                 .shape = noise_shape,
                                 .ndim = 4,
                                 .type = DT_AI_FLOAT};
  }

  int64_t output_shape[] = {1, 3, h, w};
  dt_ai_tensor_t output
    = {.data = (void *)out_patch, .shape = output_shape, .ndim = 4, .type = DT_AI_FLOAT};

  int ret = dt_ai_run(ctx, inputs, num_inputs, &output, 1);
  g_free(noise_map);
  if(ret != 0)
    return ret;

  // Undo brightness boost on the output, then convert sRGB -> linear
  const float inv_boost = 1.0f / boost;
  for(int i = 0; i < total_pixels; i++)
  {
    float v = out_patch[i] * inv_boost;
    out_patch[i] = _srgb_to_linear(v);
  }

  return 0;
}

static inline int _mirror(int v, int max)
{
  if(v < 0)
    v = -v;
  if(v >= max)
    v = 2 * max - 2 - v;
  if(v < 0)
    return 0;
  if(v >= max)
    return max - 1;
  return v;
}

/**
 * @brief Select the largest tile size that fits within available memory.
 *
 * Candidate sizes are all divisible by 64 (common neural-net alignment).
 * We reserve 1/4 of darktable's configured available memory for tile buffers
 * and ONNX Runtime intermediates.
 */
static int _select_tile_size(int num_inputs)
{
  static const int candidates[] = {2048, 1536, 1024, 768, 512, 384, 256};
  static const int n_candidates = sizeof(candidates) / sizeof(candidates[0]);

  const size_t avail = dt_get_available_mem();
  const size_t budget = avail / 4;

  for(int i = 0; i < n_candidates; i++)
  {
    const size_t T = (size_t)candidates[i];
    // tile_in + tile_out (planar float32, 3 channels each)
    const size_t tile_bufs = T * T * 3 * sizeof(float) * 2;
    // noise map for multi-input models (e.g. FFDNet)
    const size_t noise_buf = (num_inputs >= 2) ? T * T * sizeof(float) : 0;
    // Estimate for ONNX Runtime internal activations.
    // UNet/NAFNet architectures keep encoder feature maps alive for skip
    // connections while running the decoder. Peak memory scales with
    // T^2 * model_width * num_levels. Empirically ~100x the raw input tensor.
    const size_t ort_overhead = T * T * 3 * sizeof(float) * 100;
    const size_t total = tile_bufs + noise_buf + ort_overhead;

    if(total <= budget)
    {
      dt_print(
        DT_DEBUG_AI,
        "[denoise_ai] Tile size %d selected (need %zuMB, budget %zuMB)",
        candidates[i],
        total / (1024 * 1024),
        budget / (1024 * 1024));
      return candidates[i];
    }
  }

  // Fallback to smallest candidate
  dt_print(
    DT_DEBUG_AI,
    "[denoise_ai] Using minimum tile size %d (budget %zuMB)",
    candidates[n_candidates - 1],
    budget / (1024 * 1024));
  return candidates[n_candidates - 1];
}

static int _process_tiled(
  dt_ai_context_t *ctx,
  const float *in_data,
  int width,
  int height,
  TIFF *tif,
  dt_job_t *control_job,
  float sigma,
  int tile_size)
{
  const int T = tile_size;
  const int O = 64;
  const int step = T - 2 * O;
  const size_t tile_plane_size = (size_t)T * T;
  const size_t tile_buf_size = tile_plane_size * 3 * sizeof(float);

  const int cols = (width + step - 1) / step;
  const int rows = (height + step - 1) / step;
  const int total_tiles = cols * rows;

  dt_print(
    DT_DEBUG_AI,
    "[denoise_ai] Tiling %dx%d image into %dx%d grid (%d tiles, tile=%d, overlap=%d)",
    width,
    height,
    cols,
    rows,
    total_tiles,
    T,
    O);

  int res = 0;
  int tile_count = 0;

  float *tile_in = g_try_malloc(tile_buf_size);
  float *tile_out = g_try_malloc(tile_buf_size);
  // Row buffer: holds one tile-row of output scanlines (step scanlines)
  float *row_buf = g_try_malloc((size_t)width * step * 3 * sizeof(float));
  if(!tile_in || !tile_out || !row_buf)
  {
    g_free(tile_in);
    g_free(tile_out);
    g_free(row_buf);
    return 1;
  }

  for(int ty = 0; ty < rows; ty++)
  {
    const int y = ty * step;
    const int valid_h = (y + step > height) ? height - y : step;

    memset(row_buf, 0, (size_t)width * valid_h * 3 * sizeof(float));

    for(int tx = 0; tx < cols; tx++)
    {
      if(dt_control_job_get_state(control_job) == DT_JOB_STATE_CANCELLED)
      {
        dt_print(
          DT_DEBUG_AI,
          "[denoise_ai] Cancelled at tile %d/%d",
          tile_count,
          total_tiles);
        res = 1;
        goto cleanup;
      }

      const int x = tx * step;
      const int in_x_start = x - O;
      const int in_y_start = y - O;

      // Check if this tile needs border mirroring
      const int needs_mirror
        = (in_x_start < 0 || in_y_start < 0 || in_x_start + T > width || in_y_start + T > height);

      // Extract patch: interleaved RGBx -> planar RGB
      if(needs_mirror)
      {
        for(int dy = 0; dy < T; ++dy)
        {
          const int src_y = _mirror(in_y_start + dy, height);
          for(int dx = 0; dx < T; ++dx)
          {
            const int src_x = _mirror(in_x_start + dx, width);
            const size_t pixel_offset = (size_t)dy * T + dx;
            const size_t idx = ((size_t)src_y * width + src_x) * 4;
            tile_in[pixel_offset] = in_data[idx + 0];
            tile_in[pixel_offset + tile_plane_size] = in_data[idx + 1];
            tile_in[pixel_offset + 2 * tile_plane_size] = in_data[idx + 2];
          }
        }
      }
      else
      {
        // Fast path: no mirroring needed, row-sequential access
        for(int dy = 0; dy < T; ++dy)
        {
          const int src_y = in_y_start + dy;
          const float *row = in_data + (size_t)src_y * width * 4 + in_x_start * 4;
          const size_t row_offset = (size_t)dy * T;
          for(int dx = 0; dx < T; ++dx)
          {
            tile_in[row_offset + dx] = row[dx * 4 + 0];
            tile_in[row_offset + dx + tile_plane_size] = row[dx * 4 + 1];
            tile_in[row_offset + dx + 2 * tile_plane_size] = row[dx * 4 + 2];
          }
        }
      }

      // Run inference
      if(_run_patch(ctx, tile_in, T, T, tile_out, tile_count, sigma) != 0)
      {
        dt_print(DT_DEBUG_AI, "[denoise_ai] Inference failed for tile %d,%d", x, y);
        res = 1;
        goto cleanup;
      }

      // Write valid region to row buffer (excluding overlap)
      const int valid_w = (x + step > width) ? width - x : step;

      for(int dy = 0; dy < valid_h; ++dy)
      {
        const size_t tile_row = (size_t)(O + dy) * T + O;
        const size_t dst_row = ((size_t)dy * width + x) * 3;
        for(int dx = 0; dx < valid_w; ++dx)
        {
          row_buf[dst_row + dx * 3 + 0] = tile_out[tile_row + dx];
          row_buf[dst_row + dx * 3 + 1] = tile_out[tile_row + dx + tile_plane_size];
          row_buf[dst_row + dx * 3 + 2] = tile_out[tile_row + dx + 2 * tile_plane_size];
        }
      }

      tile_count++;
      if(control_job)
      {
        dt_control_job_set_progress(control_job, (double)tile_count / total_tiles);
      }
    }

    // Flush completed tile row to TIFF as 32-bit float scanlines.
    for(int dy = 0; dy < valid_h; dy++)
    {
      float *src = row_buf + (size_t)dy * width * 3;
      if(TIFFWriteScanline(tif, src, y + dy, 0) < 0)
      {
        dt_print(DT_DEBUG_AI, "[denoise_ai] TIFF write error at scanline %d", y + dy);
        res = 1;
        goto cleanup;
      }
    }
  }

cleanup:
  g_free(tile_in);
  g_free(tile_out);
  g_free(row_buf);
  return res;
}

static int _ai_write_image(
  dt_imageio_module_data_t *data,
  const char *filename,
  const void *in_void,
  dt_colorspaces_color_profile_type_t over_type,
  const char *over_filename,
  void *exif,
  int exif_len,
  dt_imgid_t imgid,
  int num,
  int total,
  dt_dev_pixelpipe_t *pipe,
  const gboolean export_masks)
{
  dt_denoise_format_params_t *params = (dt_denoise_format_params_t *)data;
  dt_denoise_job_t *job = params->job;

  // Load model if it was released after a previous image
  if(!job->ctx)
  {
    dt_print(DT_DEBUG_AI, "[denoise_ai] Reloading model for next image");
    job->ctx = dt_ai_load_model(job->env, job->model_id, NULL, job->provider);
  }
  if(!job->ctx)
    return 1;

  const int width = params->parent.width;
  const int height = params->parent.height;
  const float *in_data = (const float *)in_void;

  dt_print(DT_DEBUG_AI, "[denoise_ai] Processing image %dx%d", width, height);

  // Open TIFF before tiled processing so scanlines can be streamed
  // directly to disk, avoiding a full-resolution output buffer.
#ifdef _WIN32
  wchar_t *wfilename = g_utf8_to_utf16(filename, -1, NULL, NULL, NULL);
  TIFF *tif = TIFFOpenW(wfilename, "w");
  g_free(wfilename);
#else
  TIFF *tif = TIFFOpen(filename, "w");
#endif
  if(!tif)
  {
    dt_control_log(_("failed to open TIFF for writing: %s"), filename);
    return 1;
  }

  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3);
  TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, 0));

  const int num_inputs = dt_ai_get_input_count(job->ctx);
  const int tile_size = _select_tile_size(num_inputs);

  int res = _process_tiled(
    job->ctx,
    in_data,
    width,
    height,
    tif,
    job->control_job,
    job->sigma,
    tile_size);

  TIFFClose(tif);

  // Release ONNX model to free runtime memory.
  // The model will be reloaded if another image needs processing.
  dt_ai_unload_model(job->ctx);
  job->ctx = NULL;

  if(res != 0)
    g_unlink(filename);

  return res;
}

static void _import_image(const char *filename)
{
  dt_film_t film;
  dt_film_init(&film);
  char *dir = g_path_get_dirname(filename);
  dt_filmid_t filmid = dt_film_new(&film, dir);
  g_free(dir);
  const dt_imgid_t newid = dt_image_import(filmid, filename, FALSE, FALSE);
  dt_film_cleanup(&film);

  if(dt_is_valid_imgid(newid))
  {
    dt_print(DT_DEBUG_AI, "[denoise_ai] Imported imgid=%d: %s", newid, filename);
    dt_collection_update_query(
      darktable.collection,
      DT_COLLECTION_CHANGE_RELOAD,
      DT_COLLECTION_PROP_UNDEF,
      NULL);
    DT_CONTROL_SIGNAL_RAISE(DT_SIGNAL_VIEWMANAGER_THUMBTABLE_ACTIVATE, newid);
  }
}

static void _update_button_sensitivity(dt_lib_denoise_ai_t *d);

static gboolean _job_finished_idle(gpointer data)
{
  dt_lib_module_t *self = (dt_lib_module_t *)data;
  dt_lib_denoise_ai_t *d = (dt_lib_denoise_ai_t *)self->data;
  if(d)
  {
    d->job_running = FALSE;
    _update_button_sensitivity(d);
  }
  return G_SOURCE_REMOVE;
}

static void _job_cleanup(void *param)
{
  dt_denoise_job_t *job = (dt_denoise_job_t *)param;
  if(job->ctx)
    dt_ai_unload_model(job->ctx);
  g_free(job->model_id);
  g_list_free(job->images);
  g_free(job);
}

static int32_t _process_job_run(dt_job_t *job)
{
  dt_denoise_job_t *j = dt_control_job_get_params(job);

  dt_control_job_set_progress_message(job, "Loading AI model...");

  j->control_job = job;
  j->ctx = dt_ai_load_model(j->env, j->model_id, NULL, j->provider);

  if(!j->ctx)
  {
    dt_control_log(_("failed to load AI model: %s"), j->model_id);
    return 1;
  }

  dt_print(
    DT_DEBUG_AI,
    "[denoise_ai] Job started: model=%s, sigma=%.1f, images=%d",
    j->model_id,
    j->sigma,
    g_list_length(j->images));

  dt_imageio_module_format_t fmt = {
    .mime = _ai_get_mime,
    .levels = _ai_check_levels,
    .bpp = _ai_check_bpp,
    .write_image = _ai_write_image};

  dt_denoise_format_params_t fmt_params = {.job = j};

  int total = g_list_length(j->images);
  int count = 0;

  GList *iter = j->images;
  while(iter)
  {
    if(dt_control_job_get_state(job) == DT_JOB_STATE_CANCELLED)
      break;

    dt_imgid_t imgid = GPOINTER_TO_INT(iter->data);
    char filename[PATH_MAX];
    dt_image_full_path(imgid, filename, sizeof(filename), NULL);

    char *ext = strrchr(filename, '.');
    if(ext)
      *ext = '\0';
    g_strlcat(filename, "_denoised.tif", sizeof(filename));

    // Avoid overwriting existing denoised files: append _1, _2, etc.
    if(g_file_test(filename, G_FILE_TEST_EXISTS))
    {
      char base[PATH_MAX];
      g_strlcpy(base, filename, sizeof(base));
      char *tif_ext = strrchr(base, '.');
      if(tif_ext)
        *tif_ext = '\0'; // strip ".tif"

      for(int suffix = 1; suffix < 10000; suffix++)
      {
        snprintf(filename, sizeof(filename), "%s_%d.tif", base, suffix);
        if(!g_file_test(filename, G_FILE_TEST_EXISTS))
          break;
      }

      if(g_file_test(filename, G_FILE_TEST_EXISTS))
      {
        dt_print(
          DT_DEBUG_AI,
          "[denoise_ai] Could not find unique filename for imgid %d",
          imgid);
        dt_control_log(_("AI denoise: too many existing output files"));
        dt_control_job_set_progress(job, (double)++count / total);
        iter = g_list_next(iter);
        continue;
      }
    }

    dt_print(DT_DEBUG_AI, "[denoise_ai] Denoising imgid %d -> %s", imgid, filename);
    dt_control_job_set_progress_message(job, "Denoising image...");

    const int export_err = dt_imageio_export_with_flags(
      imgid,
      filename,
      &fmt,
      (dt_imageio_module_data_t *)&fmt_params,
      TRUE,
      FALSE,
      TRUE,
      TRUE,
      FALSE,
      1.0,
      FALSE,
      NULL,
      FALSE,
      FALSE,
      DT_COLORSPACE_LIN_REC709,
      NULL,
      DT_INTENT_PERCEPTUAL,
      NULL,
      NULL,
      count,
      total,
      NULL,
      -1);

    if(export_err)
    {
      dt_print(DT_DEBUG_AI, "[denoise_ai] Export failed for imgid %d", imgid);
      dt_control_log(_("AI denoise export failed for image"));
      dt_control_job_set_progress(job, (double)++count / total);
      iter = g_list_next(iter);
      continue;
    }

    _import_image(filename);

    dt_control_job_set_progress(job, (double)++count / total);
    iter = g_list_next(iter);
  }

  g_idle_add(_job_finished_idle, j->self);
  return 0;
}

static void _update_button_sensitivity(dt_lib_denoise_ai_t *d)
{
  gboolean sensitive = FALSE;
  if(
    d->model_available && !d->job_running
    && dt_is_valid_imgid(darktable.develop->image_storage.id))
  {
    sensitive = TRUE;
  }
  gtk_widget_set_sensitive(d->button, sensitive);
}

static void _image_changed_callback(gpointer instance, dt_lib_module_t *self)
{
  dt_lib_denoise_ai_t *d = (dt_lib_denoise_ai_t *)self->data;
  _update_button_sensitivity(d);
}

static void _ai_models_changed_callback(gpointer instance, dt_lib_module_t *self)
{
  dt_lib_denoise_ai_t *d = (dt_lib_denoise_ai_t *)self->data;

  // Refresh the AI environment to discover newly downloaded models
  if(d->env)
    dt_ai_env_refresh(d->env);

  // Re-check model availability
  d->model_available = FALSE;
  if(d->env && d->model_id)
  {
    const dt_ai_model_info_t *info = dt_ai_get_model_info_by_id(d->env, d->model_id);
    if(info && strcmp(info->task_type, "denoise") == 0)
    {
      d->model_available = TRUE;
      dt_print(
        DT_DEBUG_AI,
        "[denoise_ai] Model now available: %s (%s)",
        info->name,
        d->model_id);
    }
  }

  _update_button_sensitivity(d);
}

static void _button_clicked(GtkWidget *widget, gpointer user_data)
{
  dt_lib_module_t *self = (dt_lib_module_t *)user_data;
  dt_lib_denoise_ai_t *d = (dt_lib_denoise_ai_t *)self->data;
  dt_imgid_t imgid = darktable.develop->image_storage.id;

  if(d->model_available && !d->job_running && imgid != -1)
  {
    dt_denoise_job_t *job_data = g_new0(dt_denoise_job_t, 1);
    job_data->env = d->env;
    job_data->model_id = g_strdup(d->model_id);
    job_data->images = g_list_append(NULL, GINT_TO_POINTER(imgid));
    job_data->sigma = dt_bauhaus_slider_get(d->sigma_slider);
    job_data->provider = darktable.ai_registry->provider;
    job_data->self = self;

    d->job_running = TRUE;
    _update_button_sensitivity(d);

    dt_job_t *job = dt_control_job_create(_process_job_run, "ai denoise");
    dt_control_job_set_params(job, job_data, _job_cleanup);
    dt_control_job_add_progress(job, "Processing...", TRUE);
    dt_control_add_job(DT_JOB_QUEUE_USER_BG, job);
  }
}

void gui_init(dt_lib_module_t *self)
{
  dt_lib_denoise_ai_t *d = g_new0(dt_lib_denoise_ai_t, 1);
  self->data = d;
  d->env = dt_ai_env_init(NULL);
  d->box = GTK_BOX(gtk_box_new(GTK_ORIENTATION_VERTICAL, 5));

  // Read model ID from config (default: nafnet-sidd-width32)
  if(!dt_conf_key_exists(CONF_MODEL_KEY))
    dt_conf_set_string(CONF_MODEL_KEY, DEFAULT_MODEL_ID);
  d->model_id = dt_conf_get_string(CONF_MODEL_KEY);

  // Check if the configured model is available
  d->model_available = FALSE;
  if(d->env && d->model_id)
  {
    const dt_ai_model_info_t *info = dt_ai_get_model_info_by_id(d->env, d->model_id);
    if(info && strcmp(info->task_type, "denoise") == 0)
    {
      d->model_available = TRUE;
      dt_print(DT_DEBUG_AI, "[denoise_ai] Using model: %s (%s)", info->name, d->model_id);
    }
    else
    {
      dt_print(
        DT_DEBUG_AI,
        "[denoise_ai] Model not found: %s (module disabled)",
        d->model_id);
    }
  }

  // Noise sigma slider (only shown for multi-input models like FFDNet)
  d->sigma_slider
    = dt_bauhaus_slider_new_action(DT_ACTION(self), 0.0, 75.0, 1.0, 25.0, 0);
  dt_bauhaus_widget_set_label(d->sigma_slider, NULL, N_("sigma"));
  gtk_widget_set_tooltip_text(
    d->sigma_slider,
    _("noise level (0-75). used by models that accept a noise strength parameter"));
  gtk_box_pack_start(d->box, d->sigma_slider, FALSE, FALSE, 0);

  // Show sigma slider only if the model needs it (num_inputs >= 2)
  gboolean show_sigma = FALSE;
  if(d->model_available && d->env)
  {
    const dt_ai_model_info_t *info = dt_ai_get_model_info_by_id(d->env, d->model_id);
    if(info && info->num_inputs >= 2)
      show_sigma = TRUE;
  }
  gtk_widget_set_no_show_all(d->sigma_slider, !show_sigma);
  gtk_widget_set_visible(d->sigma_slider, show_sigma);

  d->button = gtk_button_new_with_label(_("denoise"));
  gtk_widget_set_sensitive(d->button, FALSE);
  gtk_box_pack_start(d->box, d->button, FALSE, FALSE, 0);
  g_signal_connect(d->button, "clicked", G_CALLBACK(_button_clicked), self);
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_IMAGE_CHANGED, _image_changed_callback);
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_AI_MODELS_CHANGED, _ai_models_changed_callback);
  self->widget = GTK_WIDGET(d->box);
  gtk_widget_show_all(self->widget);
  _update_button_sensitivity(d);
}

void gui_cleanup(dt_lib_module_t *self)
{
  dt_lib_denoise_ai_t *d = (dt_lib_denoise_ai_t *)self->data;

  DT_CONTROL_SIGNAL_DISCONNECT(_image_changed_callback, self);
  DT_CONTROL_SIGNAL_DISCONNECT(_ai_models_changed_callback, self);

  if(d)
  {
    g_free(d->model_id);
    if(d->env)
      dt_ai_env_destroy(d->env);
    g_free(d);
  }
  self->data = NULL;
}

void gui_update(dt_lib_module_t *self) {}
void gui_reset(dt_lib_module_t *self) {}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
