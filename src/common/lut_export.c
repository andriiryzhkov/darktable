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

#include "common/lut_export.h"
#include "common/lut_export_colorspace.h"
#include "common/colorspaces.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/darktable.h"
#include "common/film.h"
#include "common/history.h"
#include "common/image.h"
#include "common/illuminants.h"
#include "common/image_cache.h"
#include "common/math.h"
#include "common/matrices.h"
#include "common/iop_order.h"
#include "common/styles.h"
#include "common/undo.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/pixelpipe_hb.h"

#include <glib/gstdio.h>
#include <math.h>
#include <png.h>
#include <setjmp.h>
#include <stdio.h>
#include <string.h>

// ---- color space helpers ----
//
// curves and primaries live in the registry (data/lut_export.json), so a new
// camera format is a data file rather than a patch. only the math the
// exporter layers on top stays here

static inline float _encode(const dt_lut_cs_t *cs, const float lin)
{
  return dt_lut_cs_encode(cs, lin);
}

static inline float _decode(const dt_lut_cs_t *cs, const float v)
{
  return dt_lut_cs_decode(cs, v);
}

static const char *_cs_name(const dt_lut_cs_t *cs)
{
  return (cs && cs->name) ? cs->name : "?";
}

// the pipe's fixed working reference. colorin and colorout are both pinned
// to linear Rec.709, so every gamut rotation the exporter does passes
// through these primaries
static const dt_lut_cs_t *_pipe_gamut(void)
{
  return dt_lut_cs_find("linear-rec709");
}

// ---- primaries ----
//
// converting the transfer curve without the primaries leaves saturated
// colors wrong by a fixed amount no exposure tuning can fix.
//
// a wider source gamut legitimately produces negative values near its
// primaries — V-Gamut's blue y is itself negative. the pipe carries them and
// the writer clamps them

// build the matrix taking linear RGB in `from` primaries to linear RGB in
// `to` primaries, via XYZ. returns FALSE when the two describe the same
// gamut and the caller should skip the conversion entirely rather than
// multiply by a matrix that is only approximately the identity
static gboolean _make_gamut_matrix(const dt_lut_cs_t *from,
                                   const dt_lut_cs_t *to,
                                   dt_colormatrix_t matrix)
{
  if(!from || !to) return FALSE;

  if(!memcmp(from->primaries, to->primaries, sizeof(from->primaries))
     && !memcmp(from->whitepoint, to->whitepoint, sizeof(from->whitepoint)))
    return FALSE;

  dt_colormatrix_t src_to_xyz, dst_to_xyz, xyz_to_dst;
  dt_make_transposed_matrices_from_primaries_and_whitepoint
    (from->primaries, from->whitepoint, src_to_xyz);
  dt_make_transposed_matrices_from_primaries_and_whitepoint
    (to->primaries, to->whitepoint, dst_to_xyz);

  if(mat3SSEinv(xyz_to_dst, dst_to_xyz))
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] cannot invert primaries matrix for %s",
             _cs_name(to));
    return FALSE;
  }

  // same composition order as iop/primaries.c:118
  dt_colormatrix_mul(matrix, src_to_xyz, xyz_to_dst);
  return TRUE;
}

// ---- report ----

void dt_lut_export_report_free(dt_lut_export_report_t *report)
{
  if(!report) return;
  g_free(report->skipped);
  g_free(report->adjusted);
  g_free(report->range);
  memset(report, 0, sizeof(*report));
}

gboolean dt_lut_export_write_and_report(const float *rgb,
                                        const int grid_size,
                                        const char *title,
                                        const char *filepath,
                                        const dt_lut_export_report_t *report)
{
  const gboolean ok =
    rgb && dt_lut_export_write_cube(rgb, grid_size, title, filepath);

  // a NULL grid means the render failed, not the write
  if(ok)
    dt_control_log(_("exported LUT %s"), filepath);
  else if(rgb)
    dt_control_log(_("failed to write %s"), filepath);
  else
    dt_control_log(_("`%s' could not be rendered as a LUT"), title);

  if(report)
  {
    // name the file written, not its source: that is what tells a batch
    // export's messages apart
    gchar *name = g_path_get_basename(filepath);

    if(report->skipped)
      dt_control_log(_("exported LUT %s:\nskipped modules that cannot be"
                       " baked into a LUT: %s"), name, report->skipped);

    if(report->adjusted)
      dt_control_log(_("exported LUT %s:\nbaked with adjustments: %s"),
                     name, report->adjusted);

    // several clauses long; break it where it was joined so the toast fits
    if(report->range)
    {
      gchar **clauses = g_strsplit(report->range, "; ", -1);
      gchar *stacked = g_strjoinv("\n", clauses);
      dt_control_log(_("exported LUT %s:\n%s"), name, stacked);
      g_free(stacked);
      g_strfreev(clauses);
    }

    g_free(name);
  }

  return ok;
}

// the reference mid-grey both filmic and sigmoid are anchored on
#define DT_LUT_MIDDLE_GREY 0.1845f

// how much highlight range the input axis carries, in stops over mid-grey.
// this is what a style has to be built for: an axis reaching +8 EV needs a
// tone mapper configured for +8 EV
static float _input_headroom_ev(const dt_lut_cs_t *cs)
{
  const float top = _decode(cs, 1.0f);
  return (top > DT_LUT_MIDDLE_GREY)
    ? log2f(top / DT_LUT_MIDDLE_GREY)
    : 0.0f;
}

// ---- .cube writer ----

gboolean dt_lut_export_write_cube(const float *rgb,
                                  const int grid_size,
                                  const char *title,
                                  const char *filepath)
{
  if(!rgb || !filepath || grid_size < 2 || grid_size > 256) return FALSE;

  FILE *f = g_fopen(filepath, "w");
  if(!f)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] cannot open %s for writing", filepath);
    return FALSE;
  }

  // header: TITLE (optional), grid size, domain
  if(title && title[0])
    fprintf(f, "TITLE \"%s\"\n", title);
  fprintf(f, "LUT_3D_SIZE %d\n", grid_size);
  fprintf(f, "DOMAIN_MIN 0.0 0.0 0.0\n");
  fprintf(f, "DOMAIN_MAX 1.0 1.0 1.0\n\n");

  // .cube entries are clamped to the [0,1] output domain: LUT engines in
  // cameras and NLEs either clip or reject anything outside it. the callers
  // that care about how much was lost get it diagnosed in detail from
  // dt_lut_export_render_style's report
  const size_t n = (size_t)grid_size * grid_size * grid_size;
  size_t clipped = 0;

  for(size_t i = 0; i < n; i++)
  {
    float v[3];
    for(int c = 0; c < 3; c++)
    {
      const float raw = rgb[3 * i + c];
      v[c] = CLAMPF(raw, 0.0f, 1.0f);
      if(raw != v[c]) clipped++;
    }
    fprintf(f, "%.6f %.6f %.6f\n", v[0], v[1], v[2]);
  }

  fclose(f);

  if(clipped)
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] %s: clamped %zu of %zu channel values to [0,1]",
             filepath, clipped, n * 3);

  return TRUE;
}

// ---- ephemeral reference image ----
//
// the shadow pipe needs a real dt_image_t as metadata carrier (colorin and
// friends do dt_image_cache_get(dev->image_storage.id) — a DB lookup on a
// synthetic id would crash). rather than depend on the user's selection,
// we import a tiny placeholder PPM in and out of the DB around each render.
// raise_signals=FALSE prevents the catalog UI from ever seeing the entry

typedef struct _lut_ref_t
{
  dt_imgid_t imgid;
  dt_filmid_t filmid;
  gchar *dir;
  gchar *path;
} _lut_ref_t;

// 16-bit PNG identity hald, (grid_size^2) wide by grid_size tall, with pixel
// (x, y) holding LUT coordinate (r = y, g = x/grid_size, b = x%grid_size) so
// the raster index equals the .cube index r*N^2 + g*N + b.
//
// the pipe never reads these pixels — _build_pipe_input supplies the real
// input. the file exists only to give DT a loadable image to hang metadata
// off, and carries the pattern so it means something if inspected.
//
// PNG rather than PNM because dt_image_import runs dt_exif_read on
// everything, and exiv2 has no PNM reader, so every export logged a warning
static gboolean _write_identity_hald_png(const char *path, const int grid_size)
{
  const int w = grid_size * grid_size;
  const int h = grid_size;

  // allocated before setjmp: a local modified afterwards would be
  // indeterminate on the error path
  guint8 *row = g_try_malloc((size_t)w * 3 * 2);
  if(!row) return FALSE;

  FILE *f = g_fopen(path, "wb");
  if(!f)
  {
    g_free(row);
    return FALSE;
  }

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                            NULL, NULL, NULL);
  png_infop info = png ? png_create_info_struct(png) : NULL;

  // short-circuits before setjmp when png is NULL
  if(!png || !info || setjmp(png_jmpbuf(png)))
  {
    if(png) png_destroy_write_struct(&png, info ? &info : NULL);
    fclose(f);
    g_free(row);
    return FALSE;
  }

  png_init_io(png, f);
  png_set_IHDR(png, info, w, h, 16, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  // the pixels are never read back, so spend no time compressing them
  png_set_compression_level(png, 1);
  png_write_info(png, info);

  const float step = 65535.0f / (float)(grid_size - 1);
  for(int y = 0; y < h; y++)
  {
    const guint16 r_val = (guint16)(y * step + 0.5f);
    for(int x = 0; x < w; x++)
    {
      const int g_idx = x / grid_size;
      const int b_idx = x % grid_size;
      const guint16 g_val = (guint16)(g_idx * step + 0.5f);
      const guint16 b_val = (guint16)(b_idx * step + 0.5f);
      // 16-bit PNG samples are big-endian; packing the bytes by hand keeps
      // this independent of the host's endianness
      row[6 * x + 0] = (guint8)(r_val >> 8);
      row[6 * x + 1] = (guint8)(r_val & 0xff);
      row[6 * x + 2] = (guint8)(g_val >> 8);
      row[6 * x + 3] = (guint8)(g_val & 0xff);
      row[6 * x + 4] = (guint8)(b_val >> 8);
      row[6 * x + 5] = (guint8)(b_val & 0xff);
    }
    png_write_row(png, row);
  }

  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  fclose(f);
  g_free(row);
  return TRUE;
}

static gboolean _lut_ref_setup(const int grid_size, _lut_ref_t *ref)
{
  memset(ref, 0, sizeof(*ref));
  ref->imgid = NO_IMGID;
  ref->filmid = -1;

  // unique temp dir under $TMPDIR: makes orphan cleanup possible and
  // isolates parallel exports
  GError *err = NULL;
  ref->dir = g_dir_make_tmp("dt_lut_XXXXXX", &err);
  if(!ref->dir)
  {
    dt_print(DT_DEBUG_ALWAYS, "[lut_export] tmp dir: %s",
             err ? err->message : "(unknown)");
    g_clear_error(&err);
    return FALSE;
  }

  ref->path = g_build_filename(ref->dir, "hald.png", NULL);
  if(!_write_identity_hald_png(ref->path, grid_size)) goto fail;

  // dt_film_t owns a mutex that must be init-zeroed before dt_film_new,
  // else SIGKILL on macOS (os_unfair_lock corrupt). see project memory
  dt_film_t film;
  dt_film_init(&film);
  ref->filmid = dt_film_new(&film, ref->dir);
  dt_film_cleanup(&film);
  if(ref->filmid <= 0) goto fail;

  // raise_signals=FALSE: no DT_SIGNAL_IMAGE_IMPORT, catalog UI never sees
  ref->imgid = dt_image_import(ref->filmid, ref->path, FALSE, FALSE);
  if(!dt_is_valid_imgid(ref->imgid)) goto fail;

  return TRUE;

fail:
  if(ref->path)
  {
    g_unlink(ref->path);
    g_free(ref->path);
    ref->path = NULL;
  }
  if(ref->dir)
  {
    g_rmdir(ref->dir);
    g_free(ref->dir);
    ref->dir = NULL;
  }
  return FALSE;
}

static void _lut_ref_teardown(_lut_ref_t *ref)
{
  if(dt_is_valid_imgid(ref->imgid))
    dt_image_remove(ref->imgid);
  // film is left to dt_film_remove_empty on next housekeeping pass;
  // dt_film_remove hits the DB harder than we need here
  if(ref->path)
  {
    g_unlink(ref->path);
    g_free(ref->path);
  }
  if(ref->dir)
  {
    g_rmdir(ref->dir);
    g_free(ref->dir);
  }
  memset(ref, 0, sizeof(*ref));
}

// ---- LUT-representable module filter ----
//
// a 3D LUT is a pure per-pixel color map, so only modules with no dependence
// on neighbouring pixels, position or image statistics can be baked. a
// whitelist rather than a blacklist: skipping an unknown module costs a
// warning, baking one costs a silently corrupt LUT.
//
// the non-obvious entries:
//   temperature   excluded on semantics, not representability — its
//                 coefficients are sensor-space and we feed a Rec.709 grid
//   monochrome    bilateral filter on its output (monochrome.c:225)
//   zonesystem    gaussian (zonesystem.c:208)
//   exposure      included: its deflicker mode reads a histogram, but
//                 commit_params only enables that for raw uint16 input
//                 (exposure.c:645) and our reference image is a PNG
//   colorharmonizer, relight   left out pending a closer look, not known-unsafe
static gboolean _op_is_lut_safe(const char *op)
{
  static const char *const safe[] =
  {
    "agx", "basecurve", "channelmixer", "channelmixerrgb", "colisa",
    "colorbalance", "colorbalancergb", "colorchecker", "colorcontrast",
    "colorcorrection", "colorequal", "colorize", "colorzones", "exposure",
    "filmic", "filmicrgb", "invert", "levels", "lowlight", "lut3d",
    "negadoctor", "primaries", "profile_gamma", "rgbcurve", "rgblevels",
    "sigmoid", "splittoning", "tonecurve", "velvia", "vibrance", NULL
  };

  for(int i = 0; safe[i]; i++)
    if(!strcmp(op, safe[i])) return TRUE;

  return FALSE;
}

// a module can be per-pixel except for one option that acts spatially.
// dropping it outright would throw away a grade a LUT carries fine, so turn
// that option off instead.
//
// colorequal is the case: the grading is per-hue HSB lookups, and every
// spatial part sits behind its `use_filter` parameter, which only suppresses
// noise and edge artifacts a hald grid does not have.
//
// editing params in place is only sound when the item was serialised by the
// version we introspect against; on a mismatch the module is dropped rather
// than guessed at
static gboolean _neutralize_spatial_params(dt_develop_t *dev,
                                           dt_style_item_t *item,
                                           GString *adjusted)
{
  if(strcmp(item->operation, "colorequal")) return TRUE;

  const dt_iop_module_t *module =
    dt_iop_get_module_from_list(dev->iop, item->operation);

  if(!module || !module->so || !module->so->get_p
     || item->module_version != module->version()
     || item->params_size != module->params_size)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] %s: cannot introspect params (version %d vs %d),"
             " dropping", item->operation, item->module_version,
             module ? module->version() : -1);
    return FALSE;
  }

  gboolean *use_filter = module->so->get_p(item->params, "use_filter");
  if(!use_filter) return FALSE;

  if(*use_filter)
  {
    *use_filter = FALSE;
    if(adjusted->len) g_string_append(adjusted, ", ");
    g_string_append(adjusted, "colorequal (guided filter off)");
  }

  return TRUE;
}

static gboolean _op_is_mandatory(const char *op)
{
  static const char *const mandatory[] =
  {
    "colorin", "colorout", "gamma", "demosaic", "rawprepare", "finalscale",
    "flip", NULL
  };

  for(int i = 0; mandatory[i]; i++)
    if(!strcmp(op, mandatory[i])) return TRUE;

  return FALSE;
}

// modules whose removal costs the user nothing, and so are not worth naming
// in a report: the mandatory ones above, plus modules that only ever act on
// raw sensor data. the exporter feeds a synthetic non-raw grid, so the pipe
// never runs the latter and dropping them changes no pixel.
//
// deliberately NOT here: temperature, which is dropped on semantics rather
// than inertness and which the user does want to know about; and the
// RGB-domain denoisers, which would have acted had they been kept
static gboolean _op_is_silent_skip(const char *op)
{
  static const char *const inert[] =
  {
    "cacorrect", "cacorrectrgb", "highlights", "hotpixels", "overexposed",
    "rawdenoise", "rawoverexposed", NULL
  };

  if(_op_is_mandatory(op)) return TRUE;

  for(int i = 0; inert[i]; i++)
    if(!strcmp(op, inert[i])) return TRUE;

  return FALSE;
}

// accumulate an operation name for the user-facing report, once each.
//
// `seen` owns its keys, so callers may pass a transient or about-to-be-freed
// string. create it with _report_seen_new
static void _report_op(GString *list, GHashTable *seen, const char *op)
{
  if(_op_is_silent_skip(op) || g_hash_table_contains(seen, op)) return;

  g_hash_table_add(seen, g_strdup(op));
  if(list->len) g_string_append(list, ", ");
  g_string_append(list, op);
}

static GHashTable *_report_seen_new(void)
{
  return g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
}

// some modules read the image at commit_params time, not just their params,
// and our reference image does not have that data. a separate axis from
// spatial-vs-per-pixel and just as damaging — the exposure bias case cost a
// full stop with nothing in the log.
//
// channelmixerrgb is the remaining one: with illuminant "as shot in camera"
// it resolves chromaticity from raw white balance coefficients
// (illuminants.h:303), which fails on a non-raw, so the stored x/y are used.
// those are normally correct, and stale only if the white balance changed
// afterwards. re-deriving would need the source image's chroma state, and
// guessing it could replace a correct value with a worse one — so report
// rather than act
static void _note_image_dependent_params(const dt_iop_module_t *module,
                                         const dt_iop_params_t *params,
                                         GString *notes)
{
  if(!module || !module->so || !module->so->get_p
     || !dt_iop_module_is(module, "channelmixerrgb"))
    return;

  const int *illuminant = module->so->get_p(params, "illuminant");
  if(!illuminant || *illuminant != DT_ILLUMINANT_CAMERA) return;

  static const char *const note =
    "channelmixerrgb (camera illuminant taken from the saved chromaticity,"
    " not re-detected)";

  // a module can appear several times in a history; say it once
  if(strstr(notes->str, note)) return;

  if(notes->len) g_string_append(notes, ", ");
  g_string_append(notes, note);
}

// drawn and raster masks make an otherwise per-pixel module vary with
// position, so they disqualify it. parametric masks key off the pixel's own
// color and stay representable
static gboolean _item_is_lut_safe(const dt_style_item_t *item)
{
  if(!item->enabled || !_op_is_lut_safe(item->operation)) return FALSE;

  if(item->blendop_params
     && item->blendop_version == dt_develop_blend_version()
     && item->blendop_params_size == sizeof(dt_develop_blend_params_t))
  {
    const dt_develop_blend_params_t *bp = item->blendop_params;
    if(bp->mask_mode & (DEVELOP_MASK_MASK | DEVELOP_MASK_RASTER)) return FALSE;
  }

  return TRUE;
}

// ---- style application ----
//
// mirrors the style branch of dt_imageio_export_with_flags (imageio.c:1124)
// with appending=FALSE, so the LUT captures the style alone rather than
// whatever DT wrote for the placeholder. returns FALSE when nothing survived
// the filter: an all-identity LUT is worse than no file
static gboolean _apply_style(dt_develop_t *dev,
                             const char *style_name,
                             dt_lut_export_report_t *report)
{
  GList *style_items = dt_styles_get_item_list(style_name, FALSE, -1, TRUE);
  if(!style_items)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] cannot find the style '%s'", style_name);
    return FALSE;
  }

  // filter before anything else touches dev: the iop-order merge below and
  // dt_ioppr_update_for_style_items must only ever see items we will apply
  GString *dropped = g_string_new(NULL);
  GString *adjusted = g_string_new(NULL);
  GHashTable *seen = _report_seen_new();
  int kept = 0;

  for(GList *si = style_items; si;)
  {
    GList *next = g_list_next(si);
    dt_style_item_t *item = si->data;

    const gboolean safe = _item_is_lut_safe(item)
      && _neutralize_spatial_params(dev, item, adjusted);

    if(safe)
    {
      const dt_iop_module_t *module =
        dt_iop_get_module_from_list(dev->iop, item->operation);

      // style params are raw from the database, so only introspect them
      // when they were written by the version we are about to run
      if(module && item->module_version == module->version()
         && item->params_size == module->params_size)
        _note_image_dependent_params(module, item->params, adjusted);

      kept++;
    }
    else
    {
      _report_op(dropped, seen, item->operation);
      dt_style_item_free(item);
      style_items = g_list_delete_link(style_items, si);
    }
    si = next;
  }

  if(dropped->len)
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] style '%s': skipped non-LUT modules: %s",
             style_name, dropped->str);

  if(adjusted->len)
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] style '%s': baked with adjustments: %s",
             style_name, adjusted->str);

  if(report)
  {
    if(dropped->len) report->skipped = g_strdup(dropped->str);
    if(adjusted->len) report->adjusted = g_strdup(adjusted->str);
  }
  g_string_free(dropped, TRUE);
  g_string_free(adjusted, TRUE);
  g_hash_table_destroy(seen);

  if(!kept)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] style '%s' has no LUT-representable modules",
             style_name);
    g_list_free_full(style_items, dt_style_item_free);
    return FALSE;
  }

  // replace, don't append
  dt_dev_pop_history_items_ext(dev, 0);
  dt_ioppr_update_for_style_items(dev, style_items, FALSE);

  // merge the style's iop-order with the image's multi-instances, if any
  GList *iop_list = dt_styles_module_order_list(style_name);
  if(iop_list)
  {
    GList *img_iop_order_list = dev->iop_order_list;
    GList *mi = dt_ioppr_extract_multi_instances_list(img_iop_order_list);
    if(mi) iop_list = dt_ioppr_merge_multi_instance_iop_order_list(iop_list, mi);
    dev->iop_order_list = iop_list;

    g_list_free_full(img_iop_order_list, g_free);
    g_list_free_full(mi, g_free);
  }

  GList *modules_used = NULL;

  for(GList *si = style_items; si; si = g_list_next(si))
  {
    dt_style_item_t *item = si->data;

    // params_size 0 marks an auto-init module: fill in the module defaults
    if(item->params_size == 0)
    {
      const dt_iop_module_t *module =
        dt_iop_get_module_from_list(dev->iop, item->operation);
      if(!module)
      {
        dt_print(DT_DEBUG_ALWAYS,
                 "[lut_export] cannot find module %s for style", item->operation);
        continue;
      }
      item->params = (dt_iop_params_t *)malloc(module->params_size);
      if(!item->params) continue;
      item->params_size = module->params_size;
      memcpy(item->params, module->default_params, module->params_size);
    }

    // append=FALSE throughout: history was popped to 0 above
    dt_styles_apply_style_item(dev, item, &modules_used, FALSE);
  }

  g_list_free(modules_used);
  g_list_free_full(style_items, dt_style_item_free);
  return TRUE;
}

// ---- controlled input transform ----
//
// force colorin to read the input as linear Rec.709 so the exporter owns the
// LUT's input axis. otherwise the axis is whatever profile DT picked for the
// placeholder, pinning the domain to display-referred [0,1] — and V-Log 1.0
// is ~46.1 in scene-linear.
//
// must run after _apply_style, to win over any colorin the style carries
static gboolean _force_linear_input(dt_develop_t *dev)
{
  dt_iop_module_t *colorin = NULL;
  for(GList *m = dev->iop; m; m = g_list_next(m))
  {
    dt_iop_module_t *mod = m->data;
    if(dt_iop_module_is(mod, "colorin"))
    {
      colorin = mod;
      break;
    }
  }

  if(!colorin || !colorin->so || !colorin->so->get_p)
  {
    dt_print(DT_DEBUG_ALWAYS, "[lut_export] no introspectable colorin module");
    return FALSE;
  }

  dt_colorspaces_color_profile_type_t *type =
    colorin->so->get_p(colorin->params, "type");
  char *filename = colorin->so->get_p(colorin->params, "filename");
  if(!type || !filename)
  {
    dt_print(DT_DEBUG_ALWAYS, "[lut_export] cannot read colorin parameters");
    return FALSE;
  }

  *type = DT_COLORSPACE_LIN_REC709;
  filename[0] = '\0';
  colorin->enabled = TRUE;

  dt_dev_add_history_item_ext(dev, colorin, TRUE, TRUE);
  return TRUE;
}

// ---- pipe input buffer ----
//
// grid_size^3 pixels as a (grid_size^2) x grid_size raster of 4ch float RGBA,
// laid out in .cube order (r slowest, b fastest) so the backbuf writes out
// with no reordering: i = y * w + x, r = y, g = x / grid_size, b = x % grid_size.
// values are scene-linear, decoded from input_cs and rotated into Rec.709
// primaries to match the pinned colorin
static float *_build_pipe_input(const dt_lut_cs_t *input_cs,
                                const int grid_size)
{
  const size_t npx = (size_t)grid_size * grid_size * grid_size;
  float *in = g_try_malloc0(npx * 4 * sizeof(float));
  if(!in) return NULL;

  dt_colormatrix_t to_pipe;
  const gboolean regamut =
    _make_gamut_matrix(input_cs, _pipe_gamut(), to_pipe);

  const float step = 1.0f / (float)(grid_size - 1);

  size_t i = 0;
  for(int r = 0; r < grid_size; r++)
  {
    const float rv = _decode(input_cs, r * step);
    for(int g = 0; g < grid_size; g++)
    {
      const float gv = _decode(input_cs, g * step);
      for(int b = 0; b < grid_size; b++)
      {
        // aligned temporaries: the matrix helpers want dt_aligned_pixel_t,
        // and the packed 3-of-4 buffer offsets are not guaranteed aligned
        dt_aligned_pixel_t lin = { rv, gv, _decode(input_cs, b * step), 0.0f };

        if(regamut)
        {
          dt_aligned_pixel_t conv;
          dt_apply_transposed_color_matrix(lin, to_pipe, conv);
          for(int c = 0; c < 3; c++) lin[c] = conv[c];
        }

        for(int c = 0; c < 3; c++) in[4 * i + c] = lin[c];
        i++;
      }
    }
  }

  return in;
}

// ---- range diagnosis ----
//
// counting clipped entries says a LUT is wrong but not why. the usual cause
// is a range mismatch: the axis sweeps the whole encoding, so a log target
// hands the style far more highlight than one authored against a raw ever
// sees. walking the neutral diagonal converts that back into an exposure.
//
// clipping is the obvious half. a soft shoulder — sigmoid especially — never
// reaches 1.0 and so slips past a clipping test while still crushing the
// highlights into the last few percent of output, so measure that too

// output level past which further input buys almost nothing visible
#define DT_LUT_SHOULDER 0.95f

// stops above the shoulder worth complaining about. a real tone curve always
// has some compression up there; several stops of it means the style was
// built for a different input range
#define DT_LUT_SHOULDER_EV 2.0f

// mean of the three channels at grid position k on the neutral diagonal
static float _diagonal_level(const float *rgb, const int grid_size, const int k)
{
  const size_t i =
    (size_t)k * grid_size * grid_size + (size_t)k * grid_size + k;
  return (rgb[3 * i + 0] + rgb[3 * i + 1] + rgb[3 * i + 2]) / 3.0f;
}

// stops over mid-grey that grid position k represents on the input axis
static float _code_to_ev(const dt_lut_cs_t *cs,
                         const int grid_size,
                         const int k)
{
  const float lin = _decode(cs, (float)k / (float)(grid_size - 1));
  return (lin > 0.0f) ? log2f(lin / DT_LUT_MIDDLE_GREY) : -99.0f;
}

static gchar *_analyze_range(const float *rgb,
                             const int grid_size,
                             const dt_lut_cs_t *input_cs)
{
  const size_t n = (size_t)grid_size * grid_size * grid_size;
  const float headroom = _input_headroom_ev(input_cs);
  size_t high = 0, low = 0;

  for(size_t i = 0; i < n; i++)
  {
    gboolean any_high = FALSE, any_low = FALSE;
    for(int c = 0; c < 3; c++)
    {
      if(rgb[3 * i + c] > 1.0f) any_high = TRUE;
      if(rgb[3 * i + c] < 0.0f) any_low = TRUE;
    }
    if(any_high) high++;
    if(any_low) low++;
  }

  // where the neutral ramp enters the shoulder, and where it pins outright
  int shoulder = -1, pinned = -1;
  for(int k = 0; k < grid_size; k++)
  {
    const float level = _diagonal_level(rgb, grid_size, k);
    if(shoulder < 0 && level >= DT_LUT_SHOULDER) shoulder = k;
    if(pinned < 0 && level >= 1.0f) pinned = k;
  }

  const float shoulder_ev = (shoulder > 0)
    ? _code_to_ev(input_cs, grid_size, shoulder)
    : headroom;
  const float squeezed = headroom - shoulder_ev;

  const gboolean report_compression =
    shoulder > 0 && pinned != shoulder && squeezed > DT_LUT_SHOULDER_EV;

  if(!high && !low && !report_compression) return NULL;

  // built as clauses joined by one separator, so the punctuation cannot
  // drift as conditions are added
  GString *msg = g_string_new(NULL);

  if(high || low)
  {
    if(high && low)
      g_string_append_printf(msg, "clips %.0f%% of entries to white"
                             " and %.0f%% to black",
                             100.0 * (double)high / (double)n,
                             100.0 * (double)low / (double)n);
    else if(high)
      g_string_append_printf(msg, "clips %.0f%% of entries to white",
                             100.0 * (double)high / (double)n);
    else
      g_string_append_printf(msg, "clips %.0f%% of entries to black",
                             100.0 * (double)low / (double)n);
  }

  if(report_compression)
  {
    if(msg->len) g_string_append(msg, "; ");
    g_string_append_printf
      (msg,
       "the last %.1f EV of the input lands in the top %.0f%% of output,"
       " above %s %.2f — highlights will read flat",
       squeezed, 100.0 * (1.0 - DT_LUT_SHOULDER),
       _cs_name(input_cs), (float)shoulder / (float)(grid_size - 1));
  }

  // tied to the compression finding, not to clipping. clipping alone is
  // usually saturated corners falling outside the output gamut, which
  // raising the white point does not fix — and on a display-referred target,
  // where the axis carries only a couple of stops, that advice would make
  // the result worse
  if(report_compression)
  {
    if(msg->len) g_string_append(msg, "; ");
    g_string_append_printf
      (msg,
       "the %s axis carries +%.1f EV — raise the tone mapper's white"
       " relative exposure to about +%.0f EV",
       _cs_name(input_cs), headroom, headroom);
  }

  return g_string_free(msg, FALSE);
}

// ---- pipeline backstop ----
//
// filtering the history is not enough: synch_all starts every piece at its
// module's default_enabled state before replaying history, so a module
// darktable auto-enables for this image runs even though nothing asked for
// it — corrupting the grid, and making the LUT depend on the exporting
// machine's workflow preferences. so switch those off after synch_all,
// exempting the modules the pipe cannot run without
static void _disable_unsafe_pieces(dt_dev_pixelpipe_t *pipe,
                                   dt_lut_export_report_t *report)
{
  GString *off = g_string_new(NULL);
  GHashTable *seen = _report_seen_new();

  for(GList *n = pipe->nodes; n; n = g_list_next(n))
  {
    dt_dev_pixelpipe_iop_t *piece = n->data;
    const char *op = piece->module->op;

    if(!piece->enabled || _op_is_mandatory(op) || _op_is_lut_safe(op)) continue;

    piece->enabled = FALSE;
    _report_op(off, seen, op);
  }

  g_hash_table_destroy(seen);

  if(off->len)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] disabled auto-enabled non-LUT modules: %s", off->str);

    // fold into the skipped list the caller already shows
    if(report)
    {
      if(report->skipped)
      {
        gchar *merged = g_strdup_printf("%s, %s", report->skipped, off->str);
        g_free(report->skipped);
        report->skipped = merged;
      }
      else
        report->skipped = g_strdup(off->str);
    }
  }

  g_string_free(off, TRUE);
}

// ---- shadow pixelpipe: run identity hald through DT's pipeline ----
//
// the caller has bootstrapped `dev` on the reference image and put the
// history it wants baked into it. here we pin colorin and colorout to linear
// Rec.709, push the identity grid through, and re-encode into output_cs
static float *_run_pipe(dt_develop_t *dev,
                        const dt_lut_cs_t *input_cs,
                        const dt_lut_cs_t *output_cs,
                        const int grid_size,
                        dt_lut_export_report_t *report)
{
  float *in = _build_pipe_input(input_cs, grid_size);
  if(!in) return NULL;

  const int iw = grid_size * grid_size;
  const int ih = grid_size;
  float *out = NULL;

  dt_dev_pixelpipe_t pipe;
  if(!dt_dev_pixelpipe_init_export(&pipe, iw, ih, IMAGEIO_FLOAT, FALSE))
  {
    g_free(in);
    return NULL;
  }

  if(!_force_linear_input(dev)) goto cleanup;

  // force output to linear Rec.709 regardless of the user's working profile.
  // colorout reads pipe->icc_type at synch_all time (during commit_params),
  // so this MUST be set before create_nodes. see restore.c for the rationale
  dt_dev_pixelpipe_set_icc(&pipe, DT_COLORSPACE_LIN_REC709, NULL,
                           DT_INTENT_PERCEPTUAL);

  dt_ioppr_resync_modules_order(dev);
  dt_dev_pixelpipe_set_input(&pipe, dev, in, iw, ih, 1.0f);
  dt_dev_pixelpipe_create_nodes(&pipe, dev);
  dt_dev_pixelpipe_synch_all(&pipe, dev);

  // before get_dimensions: a geometric module left enabled here would resize
  // the grid and the run would be thrown away below
  _disable_unsafe_pieces(&pipe, report);

  int pw = 0, ph = 0;
  dt_dev_pixelpipe_get_dimensions(&pipe, dev, iw, ih, &pw, &ph);
  if(pw <= 0 || ph <= 0) goto cleanup;
  pipe.processed_width = pw;
  pipe.processed_height = ph;

  // NB: process_no_gamma's return value signals "pipe altered mid-flight",
  // NOT success. check backbuf explicitly
  dt_dev_pixelpipe_process_no_gamma(&pipe, dev, 0, 0, pw, ph, 1.0f);

  const int bw = pipe.backbuf_width;
  const int bh = pipe.backbuf_height;
  if(!pipe.backbuf || bw != iw || bh != ih)
  {
    // something resized the grid despite the backstop above, so the mapping
    // between pixels and LUT coordinates is gone. refuse rather than emit a
    // wrong LUT
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] pipe produced %dx%d, expected %dx%d",
             bw, bh, iw, ih);
    goto cleanup;
  }

  // pipe.backbuf is 4ch RGBA float in the pipe's ICC target: linear Rec.709
  // primaries with a linear transfer (NOT gamma-encoded). rotate into the
  // target's primaries, then apply its transfer curve — the
  // convert_colorspace helper assumes a gamma-encoded input so we bypass it
  const size_t npx = (size_t)bw * bh;
  out = g_try_malloc(npx * 3 * sizeof(float));
  if(out)
  {
    dt_colormatrix_t from_pipe;
    const gboolean regamut =
      _make_gamut_matrix(_pipe_gamut(), output_cs, from_pipe);

    const float *src = (const float *)pipe.backbuf;
    for(size_t i = 0; i < npx; i++)
    {
      dt_aligned_pixel_t lin =
        { src[4 * i + 0], src[4 * i + 1], src[4 * i + 2], 0.0f };

      if(regamut)
      {
        dt_aligned_pixel_t conv;
        dt_apply_transposed_color_matrix(lin, from_pipe, conv);
        for(int c = 0; c < 3; c++) lin[c] = conv[c];
      }

      for(int c = 0; c < 3; c++) out[3 * i + c] = _encode(output_cs, lin[c]);
    }

    // diagnose before the writer clamps, while the overshoot is still there.
    // logged here as well as shown in the UI, to match how the skipped and
    // adjusted lists are reported
    gchar *range = _analyze_range(out, grid_size, input_cs);

    if(range)
    {
      // one clause per line in the log — the whole diagnosis on a single
      // line runs past any sane terminal width. the report keeps the
      // one-line form, since dt_control_log shows a single line
      gchar **clauses = g_strsplit(range, "; ", -1);
      dt_print(DT_DEBUG_ALWAYS, "[lut_export] range:");
      for(int i = 0; clauses[i]; i++)
        dt_print_nts(DT_DEBUG_ALWAYS, "            - %s\n", clauses[i]);
      g_strfreev(clauses);
    }

    if(report)
      report->range = range;
    else
      g_free(range);
  }

cleanup:
  dt_dev_pixelpipe_cleanup(&pipe);
  g_free(in);
  return out;
}

float *dt_lut_export_render_style(const char *style_name,
                                  const char *input_cs_id,
                                  const char *output_cs_id,
                                  const int grid_size,
                                  dt_lut_export_report_t *report)
{
  if(report) memset(report, 0, sizeof(*report));
  if(!style_name || grid_size < 2 || grid_size > 256) return NULL;

  const dt_lut_cs_t *input_cs = dt_lut_cs_find(input_cs_id);
  const dt_lut_cs_t *output_cs = dt_lut_cs_find(output_cs_id);
  if(!input_cs || !output_cs || !_pipe_gamut()) return NULL;

  _lut_ref_t ref;
  if(!_lut_ref_setup(grid_size, &ref)) return NULL;

  // dev bootstrap: matches src/common/ai/restore.c:652-679. gui=FALSE keeps
  // it headless. load_image reads metadata + XMP history for our placeholder,
  // which _apply_style then discards
  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dt_dev_load_image(&dev, ref.imgid);

  float *out = _apply_style(&dev, style_name, report)
    ? _run_pipe(&dev, input_cs, output_cs, grid_size, report)
    : NULL;

  dt_dev_cleanup(&dev);
  _lut_ref_teardown(&ref);
  return out;
}

// ---- current edit ----
//
// same machinery, different source: copy the image's history onto the
// placeholder and bake that. dt_history_copy_and_paste_on_image flushes the
// darkroom's in-memory history first, so an unsaved edit still exports.
//
// the filter then runs over dev->history rather than style items, whose
// params have already been through legacy conversion and so need no version
// check
static gboolean _filter_history(dt_develop_t *dev,
                                dt_lut_export_report_t *report)
{
  GString *dropped = g_string_new(NULL);
  GString *adjusted = g_string_new(NULL);
  GHashTable *seen = _report_seen_new();
  int kept = 0;

  // items past history_end are redo entries that do not apply
  while(g_list_length(dev->history) > (guint)MAX(0, dev->history_end))
  {
    GList *last = g_list_last(dev->history);
    dt_dev_free_history_item(last->data);
    dev->history = g_list_delete_link(dev->history, last);
  }

  // a history holds successive snapshots of a module instance, and synch_all
  // replays them in order, so the LAST item for an instance is the state
  // that survives. safety has to be judged on that item and applied to the
  // whole instance.
  //
  // judging items independently is both misleading and wrong. misleading
  // because an instance the user masked and then unmasked would be reported
  // as skipped while its final, maskless state was in fact baked. wrong
  // because the reverse order — unmasked early, masked at the end — would
  // drop the masked final item and leave an earlier one to apply an
  // adjustment the user had since masked away
  GHashTable *unsafe = g_hash_table_new_full(g_str_hash, g_str_equal,
                                             g_free, NULL);

  for(GList *h = dev->history; h; h = g_list_next(h))
  {
    const dt_dev_history_item_t *item = h->data;

    gboolean safe = _op_is_lut_safe(item->op_name);

    if(safe && item->blend_params
       && (item->blend_params->mask_mode
           & (DEVELOP_MASK_MASK | DEVELOP_MASK_RASTER)))
      safe = FALSE;

    // later items replace earlier verdicts for the same instance
    gchar *key = g_strdup_printf("%s.%d", item->op_name, item->multi_priority);
    if(safe)
      g_hash_table_remove(unsafe, key);
    else
      g_hash_table_replace(unsafe, g_strdup(key),
                           GINT_TO_POINTER(item->enabled ? 1 : 0));
    g_free(key);
  }

  for(GList *h = dev->history; h;)
  {
    GList *next = g_list_next(h);
    dt_dev_history_item_t *item = h->data;

    gchar *key = g_strdup_printf("%s.%d", item->op_name, item->multi_priority);
    gpointer verdict = NULL;
    const gboolean drop =
      g_hash_table_lookup_extended(unsafe, key, NULL, &verdict);
    g_free(key);

    if(!drop)
    {
      // history params have already been through legacy conversion, so they
      // always match the running module and need no version check
      _note_image_dependent_params(item->module, item->params, adjusted);
      kept++;
    }
    else
    {
      // only worth reporting a module whose surviving state was switched on
      if(GPOINTER_TO_INT(verdict)) _report_op(dropped, seen, item->op_name);
      dt_dev_free_history_item(item);
      dev->history = g_list_delete_link(dev->history, h);
    }
    h = next;
  }

  g_hash_table_destroy(unsafe);

  dev->history_end = g_list_length(dev->history);

  if(dropped->len)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] edit: skipped non-LUT modules: %s", dropped->str);
    if(report) report->skipped = g_strdup(dropped->str);
  }

  if(adjusted->len)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] edit: baked with caveats: %s", adjusted->str);
    if(report) report->adjusted = g_strdup(adjusted->str);
  }

  g_string_free(dropped, TRUE);
  g_string_free(adjusted, TRUE);
  g_hash_table_destroy(seen);

  return kept > 0;
}

float *dt_lut_export_render_image(const dt_imgid_t imgid,
                                  const char *input_cs_id,
                                  const char *output_cs_id,
                                  const int grid_size,
                                  dt_lut_export_report_t *report)
{
  if(report) memset(report, 0, sizeof(*report));
  if(!dt_is_valid_imgid(imgid) || grid_size < 2 || grid_size > 256) return NULL;

  const dt_lut_cs_t *input_cs = dt_lut_cs_find(input_cs_id);
  const dt_lut_cs_t *output_cs = dt_lut_cs_find(output_cs_id);
  if(!input_cs || !output_cs || !_pipe_gamut()) return NULL;

  _lut_ref_t ref;
  if(!_lut_ref_setup(grid_size, &ref)) return NULL;

  // the paste records a DT_UNDO_LT_HISTORY entry unconditionally, and it
  // would name a placeholder image that no longer exists by the time the
  // user could reach it with ctrl+Z. suppress that one record
  dt_undo_disable_next(darktable.undo);

  // merge=FALSE so the placeholder's own history is replaced outright,
  // sync=FALSE to skip the XMP write for an image we are about to remove
  if(!dt_history_copy_and_paste_on_image(imgid, ref.imgid, FALSE, NULL,
                                         TRUE, TRUE, FALSE))
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] cannot copy history from image %d", imgid);
    _lut_ref_teardown(&ref);
    return NULL;
  }

  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dt_dev_load_image(&dev, ref.imgid);

  // some modules read image metadata at commit_params time rather than
  // taking everything from their params. exposure is the one that bites:
  // with "compensate camera exposure bias" enabled it subtracts the image's
  // exif_exposure_bias (exposure.c:630), and our placeholder has no EXIF at
  // all — so the compensation would silently vanish and the LUT would sit up
  // to 5 EV away from the edit it claims to reproduce. carry the values the
  // pipe actually consults across from the real image
  dt_image_t *src = dt_image_cache_get(imgid, 'r');
  if(src)
  {
    dev.image_storage.exif_exposure_bias = src->exif_exposure_bias;
    dev.image_storage.exif_highlight_preservation = src->exif_highlight_preservation;
    dt_image_cache_read_release(src);
  }

  float *out = _filter_history(&dev, report)
    ? _run_pipe(&dev, input_cs, output_cs, grid_size, report)
    : NULL;

  if(!out && report && !report->skipped)
    dt_print(DT_DEBUG_ALWAYS,
             "[lut_export] image %d has no LUT-representable modules", imgid);

  dt_dev_cleanup(&dev);
  _lut_ref_teardown(&ref);
  return out;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
