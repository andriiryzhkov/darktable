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

#include <errno.h>
#include <poll.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static char *_handle_ping(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
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

static char *_handle_shutdown(dt_server_t *server, const dt_server_request_t *req)
{
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "shutting_down");
  json_builder_end_object(b);
  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);

  // Schedule shutdown after sending response
  server->running = FALSE;
  if(server->main_loop)
    g_main_loop_quit(server->main_loop);

  return resp;
}

static char *_handle_get_version(dt_server_t *server, const dt_server_request_t *req)
{
  (void)server;
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "version");
  json_builder_add_string_value(b, darktable_package_version);
  json_builder_end_object(b);
  JsonNode *result = json_builder_get_root(b);
  char *resp = dt_server_make_response(req->id, result);
  json_node_unref(result);
  g_object_unref(b);
  return resp;
}

static const dt_server_route_t _routes[] = {
  { "system.ping",                _handle_ping },
  { "system.shutdown",            _handle_shutdown },
  { "system.get_version",         _handle_get_version },
  { "catalog.query",              dt_server_catalog_query },
  { "catalog.get_image",          dt_server_catalog_get_image },
  { "catalog.get_thumbnail",      dt_server_catalog_get_thumbnail },
  { "catalog.get_tags",           dt_server_catalog_get_tags },
  { "catalog.get_filmrolls",      dt_server_catalog_get_filmrolls },
  { "catalog.check_imported",     dt_server_catalog_check_imported },
  { "catalog.import",             dt_server_catalog_import },
  { "catalog.copy_import",        dt_server_catalog_copy_import },
  { "catalog.get_file_thumbnail", dt_server_catalog_get_file_thumbnail },
  { "catalog.get_collection_values", dt_server_catalog_get_collection_values },
  { "develop.open",               dt_server_develop_open },
  { "develop.close",              dt_server_develop_close },
  { "develop.get_modules",        dt_server_develop_get_modules },
  { "develop.get_history",        dt_server_develop_get_history },
  { "develop.get_params",         dt_server_develop_get_params },
  { "develop.set_params",         dt_server_develop_set_params },
  { "develop.commit_params",      dt_server_develop_commit_params },
  { "develop.request_preview",    dt_server_develop_request_preview },
  { "develop.delete_history",     dt_server_develop_delete_history },
  { "export.image",               dt_server_export_image },
  { NULL, NULL }
};

static char *_dispatch(dt_server_t *server, const dt_server_request_t *req)
{
  for(int i = 0; _routes[i].method; i++)
  {
    if(!strcmp(_routes[i].method, req->method))
      return _routes[i].handler(server, req);
  }
  return dt_server_make_error(req->id, DT_SERVER_ERR_METHOD,
                               "Unknown method");
}

static void _handle_client(dt_server_t *server)
{
  fprintf(stderr, "[server] client connected\n");

  while(server->running)
  {
    // Drain any pending events first
    gpointer evt;
    while((evt = g_async_queue_try_pop(server->event_queue)) != NULL)
    {
      char *event_json = (char *)evt;
      size_t event_len = strlen(event_json);
      if(!dt_server_write_frame(server->client_fd, event_json, event_len))
      {
        g_free(event_json);
        fprintf(stderr, "[server] failed to send event, client disconnected\n");
        return;
      }
      g_free(event_json);
    }

    // Use poll to wait for data with timeout (so we can drain events periodically)
    struct pollfd pfd = { .fd = server->client_fd, .events = POLLIN };
    int ready = poll(&pfd, 1, 50); // 50ms timeout

    if(ready < 0)
    {
      if(errno == EINTR) continue;
      fprintf(stderr, "[server] poll error: %s\n", strerror(errno));
      break;
    }

    if(ready == 0) continue; // timeout, loop back to drain events

    if(pfd.revents & (POLLERR | POLLHUP))
    {
      fprintf(stderr, "[server] client disconnected\n");
      break;
    }

    if(!(pfd.revents & POLLIN)) continue;

    // Read a request frame
    char *frame_buf = NULL;
    size_t frame_len = 0;
    if(!dt_server_read_frame(server->client_fd, &frame_buf, &frame_len))
    {
      fprintf(stderr, "[server] client disconnected (read failed)\n");
      break;
    }

    // Parse and dispatch
    dt_server_request_t *req = dt_server_parse_request(frame_buf, frame_len);
    g_free(frame_buf);

    if(!req)
    {
      char *err = dt_server_make_error("", DT_SERVER_ERR_PARSE, "Invalid JSON request");
      dt_server_write_frame(server->client_fd, err, strlen(err));
      g_free(err);
      continue;
    }

    fprintf(stderr, "[server] <- %s (id=%s)\n", req->method, req->id);

    char *response = _dispatch(server, req);
    dt_server_free_request(req);

    if(response)
    {
      if(!dt_server_write_frame(server->client_fd, response, strlen(response)))
      {
        g_free(response);
        fprintf(stderr, "[server] failed to send response, client disconnected\n");
        break;
      }
      g_free(response);
    }
  }
}

dt_server_t *dt_server_init(const char *socket_path)
{
  dt_server_t *server = g_new0(dt_server_t, 1);
  g_strlcpy(server->socket_path, socket_path, sizeof(server->socket_path));
  server->listen_fd = -1;
  server->client_fd = -1;
  server->event_queue = g_async_queue_new();

  // Create Unix domain socket
  server->listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if(server->listen_fd < 0)
  {
    fprintf(stderr, "[server] socket() failed: %s\n", strerror(errno));
    g_free(server);
    return NULL;
  }

  // Remove any stale socket file
  g_unlink(socket_path);

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  g_strlcpy(addr.sun_path, socket_path, sizeof(addr.sun_path));

  if(bind(server->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
  {
    fprintf(stderr, "[server] bind(%s) failed: %s\n", socket_path, strerror(errno));
    close(server->listen_fd);
    g_free(server);
    return NULL;
  }

  if(listen(server->listen_fd, 1) < 0)
  {
    fprintf(stderr, "[server] listen() failed: %s\n", strerror(errno));
    close(server->listen_fd);
    g_unlink(socket_path);
    g_free(server);
    return NULL;
  }

  fprintf(stderr, "[server] listening on %s\n", socket_path);
  return server;
}

void dt_server_run(dt_server_t *server)
{
  server->running = TRUE;

  // Initialize event bridge (connects to darktable signals)
  dt_server_events_init(server);

  // Accept loop: one client at a time (v1 simplicity)
  while(server->running)
  {
    fprintf(stderr, "[server] waiting for client...\n");

    struct sockaddr_un client_addr;
    socklen_t client_len = sizeof(client_addr);
    int cfd = accept(server->listen_fd, (struct sockaddr *)&client_addr, &client_len);

    if(cfd < 0)
    {
      if(errno == EINTR) continue;
      if(!server->running) break;
      fprintf(stderr, "[server] accept() failed: %s\n", strerror(errno));
      continue;
    }

    server->client_fd = cfd;
    _handle_client(server);

    // Client disconnected — close all develop sessions
    for(int i = 0; i < server->session_count; i++)
    {
      if(server->sessions[i])
      {
        dt_server_session_t *s = server->sessions[i];
        dt_shm_destroy(&s->shm_buffers[0]);
        dt_shm_destroy(&s->shm_buffers[1]);
        dt_dev_cleanup(&s->dev);
        g_free(s);
        server->sessions[i] = NULL;
      }
    }
    server->session_count = 0;

    close(server->client_fd);
    server->client_fd = -1;
  }

  dt_server_events_cleanup(server);
}

void dt_server_shutdown(dt_server_t *server)
{
  server->running = FALSE;
  // Close listen socket to unblock accept()
  if(server->listen_fd >= 0)
  {
    shutdown(server->listen_fd, SHUT_RDWR);
    close(server->listen_fd);
    server->listen_fd = -1;
  }
}

void dt_server_cleanup(dt_server_t *server)
{
  if(!server) return;

  if(server->client_fd >= 0)
    close(server->client_fd);
  if(server->listen_fd >= 0)
    close(server->listen_fd);

  g_unlink(server->socket_path);

  if(server->event_queue)
  {
    // Drain remaining events
    gpointer evt;
    while((evt = g_async_queue_try_pop(server->event_queue)) != NULL)
      g_free(evt);
    g_async_queue_unref(server->event_queue);
  }

  g_free(server);
}

void dt_server_queue_event(dt_server_t *server, const char *event_name, JsonNode *data)
{
  char *json = dt_server_make_event(event_name, data);
  g_async_queue_push(server->event_queue, json);
}

dt_server_session_t *dt_server_find_session(dt_server_t *server, const char *session_id)
{
  for(int i = 0; i < server->session_count; i++)
  {
    if(server->sessions[i] && !strcmp(server->sessions[i]->session_id, session_id))
      return server->sessions[i];
  }
  return NULL;
}
