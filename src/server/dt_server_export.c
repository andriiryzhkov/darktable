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

#include "server/dt_server.h"
#include "imageio/imageio_module.h"

#include <string.h>

// normalize format extension to module name (matches cli/main.c logic)
static const char *_normalize_format(const char *ext)
{
  if(!strcmp(ext, "jpg")) return "jpeg";
  if(!strcmp(ext, "tif")) return "tiff";
  if(!strcmp(ext, "jxl")) return "jpegxl";
  return ext;
}

char *dt_server_export_image(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;

  if(!req->params || !json_object_has_member(req->params, "imgid")
     || !json_object_has_member(req->params, "output_path"))
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Missing imgid or output_path parameter");

  const dt_imgid_t imgid = (dt_imgid_t)json_object_get_int_member(req->params, "imgid");
  const char *output_path = json_object_get_string_member(req->params, "output_path");

  // determine format from file extension
  const char *dot = strrchr(output_path, '.');
  if(!dot || dot == output_path)
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "output_path must have a file extension (e.g. .jpg, .png, .tiff)");

  const char *format_name = _normalize_format(dot + 1);

  // optional parameters
  int max_width = 0;
  int max_height = 0;
  gboolean high_quality = TRUE;
  gboolean upscale = FALSE;
  gboolean export_masks = FALSE;

  if(json_object_has_member(req->params, "max_width"))
    max_width = (int)json_object_get_int_member(req->params, "max_width");
  if(json_object_has_member(req->params, "max_height"))
    max_height = (int)json_object_get_int_member(req->params, "max_height");
  if(json_object_has_member(req->params, "high_quality"))
    high_quality = json_object_get_boolean_member(req->params, "high_quality");
  if(json_object_has_member(req->params, "upscale"))
    upscale = json_object_get_boolean_member(req->params, "upscale");

  // get storage module ("disk")
  dt_imageio_module_storage_t *storage = dt_imageio_get_storage_by_name("disk");
  if(!storage)
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "Cannot find disk storage module");

  dt_imageio_module_data_t *sdata = storage->get_params(storage);
  if(!sdata)
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "Failed to get storage parameters");

  // set the output filename into storage params
  // (the first field of dt_imageio_disk_t is char filename[DT_MAX_PATH_FOR_PARAMS])
  // strip the extension — the format module appends it
  char stripped_path[DT_MAX_PATH_FOR_PARAMS];
  g_strlcpy(stripped_path, output_path, sizeof(stripped_path));
  char *ext_pos = strrchr(stripped_path, '.');
  if(ext_pos) *ext_pos = '\0';

  g_strlcpy((char *)sdata, stripped_path, DT_MAX_PATH_FOR_PARAMS);

  // get format module
  dt_imageio_module_format_t *format = dt_imageio_get_format_by_name(format_name);
  if(!format)
  {
    storage->free_params(storage, sdata);
    return dt_server_make_error(req->id, DT_SERVER_ERR_PARAMS,
                                 "Unknown format for the given file extension");
  }

  dt_imageio_module_data_t *fdata = format->get_params(format);
  if(!fdata)
  {
    storage->free_params(storage, sdata);
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "Failed to get format parameters");
  }

  // set output dimensions
  fdata->max_width = max_width;
  fdata->max_height = max_height;
  fdata->style[0] = '\0';
  fdata->style_append = 1;

  // apply dimension constraints from storage/format modules
  uint32_t sw = 0, sh = 0, fw = 0, fh = 0;
  storage->dimension(storage, sdata, &sw, &sh);
  format->dimension(format, fdata, &fw, &fh);

  if(sw != 0 && fdata->max_width > (int)sw) fdata->max_width = (int)sw;
  if(sh != 0 && fdata->max_height > (int)sh) fdata->max_height = (int)sh;
  if(fw != 0 && fdata->max_width > (int)fw) fdata->max_width = (int)fw;
  if(fh != 0 && fdata->max_height > (int)fh) fdata->max_height = (int)fh;

  // initialize store if needed
  GList *id_list = g_list_append(NULL, GINT_TO_POINTER(imgid));
  if(storage->initialize_store)
  {
    storage->initialize_store(storage, sdata, &format, &fdata, &id_list, high_quality, upscale);
    format->set_params(format, fdata, format->params_size(format));
    storage->set_params(storage, sdata, storage->params_size(storage));
  }

  // set up default metadata flags
  dt_export_metadata_t metadata;
  metadata.flags = dt_lib_export_metadata_default_flags();
  metadata.list = NULL;

  fprintf(stderr, "[server] export.image: imgid=%d -> %s (format=%s)\n",
          imgid, output_path, format_name);

  // perform the export
  int result = storage->store(storage, sdata, imgid, format, fdata,
                              1, 1, high_quality, upscale, FALSE, 1.0,
                              export_masks,
                              DT_COLORSPACE_SRGB, NULL, DT_INTENT_PERCEPTUAL,
                              &metadata);

  // cleanup
  if(storage->finalize_store) storage->finalize_store(storage, sdata);
  storage->free_params(storage, sdata);
  format->free_params(format, fdata);
  g_list_free(id_list);

  if(result != 0)
    return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                                 "Export failed");

  // build response
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);

  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");

  json_builder_set_member_name(b, "imgid");
  json_builder_add_int_value(b, imgid);

  json_builder_set_member_name(b, "output_path");
  json_builder_add_string_value(b, output_path);

  json_builder_set_member_name(b, "format");
  json_builder_add_string_value(b, format_name);

  json_builder_end_object(b);

  JsonNode *res = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, res);
  json_node_unref(res);
  g_object_unref(b);
  return resp;
}
