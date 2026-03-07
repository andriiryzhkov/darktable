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

#pragma once

#include <glib.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/*
 * Validate that a filesystem path is within allowed roots.
 * Prevents directory traversal attacks from the webview JS layer.
 *
 * Allowed roots:
 *   - User home directory
 *   - Filesystem root "/" (for folder browsing navigation)
 *   - macOS: /Volumes/
 *   - Linux: /media/, /mnt/, /run/media/
 *   - Windows: any drive letter (C:\, D:\, etc.)
 *
 * Security properties:
 *   - Uses realpath() to resolve symlinks and ".." before checking
 *   - Rejects paths with "/../" when realpath() fails (non-existent paths)
 */
static inline gboolean dt_path_is_allowed(const char *path)
{
  if(!path || !path[0]) return FALSE;

  /* Resolve symlinks and normalize the path */
  char resolved[PATH_MAX];
  if(!realpath(path, resolved))
  {
    /* Path doesn't exist — validate the intent.
     * Reject paths with /../ components. */
    if(strstr(path, "/../") || (strlen(path) >= 3 && strcmp(path + strlen(path) - 3, "/..") == 0))
      return FALSE;

    /* For non-existent paths, use the path as-is for prefix checks */
    g_strlcpy(resolved, path, sizeof(resolved));
  }

  /* Always allowed: user home directory */
  const char *home = g_get_home_dir();
  if(home && g_str_has_prefix(resolved, home))
    return TRUE;

  /* Allow filesystem root itself (for folder browsing navigation) */
  if(strcmp(resolved, "/") == 0)
    return TRUE;

  /* Allow common mount points so users can navigate to external drives */
  static const char *allowed_prefixes[] = {
#ifdef __APPLE__
    "/Volumes/",
#else
    "/media/",
    "/mnt/",
    "/run/media/",
#endif
#ifdef _WIN32
    /* On Windows, all drive letters are allowed (checked below) */
#endif
    NULL
  };

  for(const char **pfx = allowed_prefixes; *pfx; pfx++)
  {
    if(g_str_has_prefix(resolved, *pfx))
      return TRUE;
  }

#ifdef _WIN32
  /* Allow any drive letter path (e.g. C:\, D:\) */
  if(((resolved[0] >= 'A' && resolved[0] <= 'Z') ||
      (resolved[0] >= 'a' && resolved[0] <= 'z')) && resolved[1] == ':')
    return TRUE;
#endif

  return FALSE;
}
