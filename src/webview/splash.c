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

#include "splash.h"

#include <glib.h>
#include <stdio.h>

// fallback template if splash.html cannot be loaded
static const char *SPLASH_FALLBACK =
    "<!DOCTYPE html><html><head><style>"
    "body{height:100vh;display:flex;align-items:center;justify-content:center;"
    "background:#3d3d3d;font-family:sans-serif;color:#e0e0e0}"
    "</style></head><body>"
    "<div style='text-align:center'>"
    "<div style='font-size:48px;font-weight:300;letter-spacing:2px;margin-bottom:32px'>darktable</div>"
    "<div id='splash-status' style='color:#999'>starting...</div>"
    "</div></body></html>";

// try to find a file relative to the binary directory
static char *_find_asset(const char *binary_dir, const char *filename)
{
  // development layout: binary at build/bin/, assets at ui/src/assets/
  char *path = g_build_filename(binary_dir, "..", "..", "ui", "src", "assets", filename, NULL);
  if(g_file_test(path, G_FILE_TEST_EXISTS)) return path;
  g_free(path);

  // installed layout: binary at bin/, data at share/darktable/splash/
  path = g_build_filename(binary_dir, "..", "share", "darktable", "splash", filename, NULL);
  if(g_file_test(path, G_FILE_TEST_EXISTS)) return path;
  g_free(path);

  return NULL;
}

// try to find splash.html relative to the binary directory
static char *_find_splash_html(const char *binary_dir)
{
  // development layout: binary at build/bin/, source at src/webview/
  char *path = g_build_filename(binary_dir, "..", "..", "src", "webview", "splash.html", NULL);
  if(g_file_test(path, G_FILE_TEST_EXISTS)) return path;
  g_free(path);

  // installed layout: binary at bin/, data at share/darktable/splash/
  path = g_build_filename(binary_dir, "..", "share", "darktable", "splash", "splash.html", NULL);
  if(g_file_test(path, G_FILE_TEST_EXISTS)) return path;
  g_free(path);

  return NULL;
}

// read file and return base64-encoded content, or NULL
static char *_load_asset_b64(const char *binary_dir, const char *filename)
{
  char *path = _find_asset(binary_dir, filename);
  if(!path) return NULL;

  gchar *data = NULL;
  gsize len = 0;
  if(!g_file_get_contents(path, &data, &len, NULL))
  {
    g_free(path);
    return NULL;
  }
  g_free(path);

  char *b64 = g_base64_encode((const guchar *)data, len);
  g_free(data);
  return b64;
}

// replace first occurrence of needle in haystack, returns new string
static char *_str_replace(const char *haystack, const char *needle, const char *replacement)
{
  const char *pos = strstr(haystack, needle);
  if(!pos) return g_strdup(haystack);

  GString *result = g_string_new_len(haystack, pos - haystack);
  g_string_append(result, replacement);
  g_string_append(result, pos + strlen(needle));
  return g_string_free(result, FALSE);
}

void dt_splash_show(webview_t w, const char *binary_dir)
{
  // load splash HTML template
  char *template_html = NULL;
  char *html_path = _find_splash_html(binary_dir);
  if(html_path)
  {
    fprintf(stderr, "[webview] splash template: %s\n", html_path);
    g_file_get_contents(html_path, &template_html, NULL, NULL);
    g_free(html_path);
  }
  if(!template_html)
  {
    fprintf(stderr, "[webview] splash.html not found, using fallback\n");
    webview_set_html(w, SPLASH_FALLBACK);
    return;
  }

  // build logo and title HTML from SVG assets
  char *logo_b64 = _load_asset_b64(binary_dir, "darktable-logo.svg");
  char *title_b64 = _load_asset_b64(binary_dir, "darktable.svg");

  char *logo_html = logo_b64
      ? g_strdup_printf("<img class='splash-logo' src='data:image/svg+xml;base64,%s'>", logo_b64)
      : g_strdup("");
  char *title_html = title_b64
      ? g_strdup_printf("<img class='splash-title' src='data:image/svg+xml;base64,%s'>", title_b64)
      : g_strdup("<span class='splash-title-text'>darktable</span>");

  // substitute placeholders
  char *tmp = _str_replace(template_html, "{{LOGO}}", logo_html);
  char *html = _str_replace(tmp, "{{TITLE}}", title_html);

  webview_set_html(w, html);

  g_free(html);
  g_free(tmp);
  g_free(template_html);
  g_free(logo_html);
  g_free(title_html);
  g_free(logo_b64);
  g_free(title_b64);
}

static void _splash_update_dispatch(webview_t w, void *arg)
{
  char *text = (char *)arg;
  char *js = g_strdup_printf(
      "document.getElementById('splash-status').textContent='%s';", text);
  webview_eval(w, js);
  g_free(js);
  g_free(text);
}

void dt_splash_update(webview_t w, const char *status)
{
  // dispatch to main thread — safe to call from any thread
  webview_dispatch(w, _splash_update_dispatch, g_strdup(status));
}
