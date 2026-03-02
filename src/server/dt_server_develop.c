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
#include "develop/develop.h"

char *dt_server_develop_open(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  // TODO: implement develop session creation
  return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                               "develop.open not yet implemented");
}

char *dt_server_develop_close(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                               "develop.close not yet implemented");
}

char *dt_server_develop_get_modules(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                               "develop.get_modules not yet implemented");
}

char *dt_server_develop_get_params(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                               "develop.get_params not yet implemented");
}

char *dt_server_develop_set_params(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                               "develop.set_params not yet implemented");
}

char *dt_server_develop_request_preview(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  return dt_server_make_error(req->id, DT_SERVER_ERR_INTERNAL,
                               "develop.request_preview not yet implemented");
}
