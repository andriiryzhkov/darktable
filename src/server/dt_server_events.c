/*
    This file is part of darktable,
    Copyright (C) 2025 darktable developers.

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

// TODO: Connect to darktable signals and forward as JSON events.
// Signals to bridge:
//   DT_SIGNAL_COLLECTION_CHANGED -> "collection.changed"
//   DT_SIGNAL_IMAGE_IMPORT -> "image.imported"
//   DT_SIGNAL_DEVELOP_MIPMAP_UPDATED -> "image.thumbnail_ready"
//   DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED -> "develop.preview_ready"
//   DT_SIGNAL_DEVELOP_HISTORY_CHANGE -> "develop.history_changed"

void dt_server_events_init(dt_server_t *server)
{
  (void)server;
  fprintf(stderr, "[server] event bridge initialized (stub)\n");
}

void dt_server_events_cleanup(dt_server_t *server)
{
  (void)server;
}
