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

#define _DEFAULT_SOURCE // for usleep with _XOPEN_SOURCE=700
#include "bindings.h"
#include "ipc.h"
#include "splash.h"
#include "titlebar.h"
#include "transport.h"

#include <errno.h>
#include <glib.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// data shared between main thread and startup thread
typedef struct _startup_data_t
{
  dt_webview_ctx_t *ctx;
  const char *server_bin;
  char **core_args;
  int core_argc;
  char *frontend_url;
} _startup_data_t;

static void usage(const char *progname)
{
  fprintf(stderr,
    "darktable-nova — webview UI for darktable\n\n"
    "Usage:\n"
    "  %s [OPTIONS] [--core DARKTABLE_OPTIONS]\n\n"
    "Options:\n"
    "  --server           Use IPC transport (connect to darktable-server)\n"
    "  --dev              Connect to Vite dev server at http://localhost:5173\n"
    "  --frontend-dir DIR Load production build from DIR (default: ui/dist)\n"
    "  --server-bin PATH  Path to darktable-server binary (implies --server)\n"
    "  -h, --help         Show this help\n\n"
    "Transport modes:\n"
    "  Default:           Direct mode (in-process libdarktable)\n"
    "  --server:          IPC mode (spawns darktable-server, communicates via socket)\n\n"
    "darktable options (after --core):\n"
    "  --configdir DIR    Config directory (default: ~/.config/darktable/)\n"
    "  --library FILE     Use specific library.db file\n"
    "  --cachedir DIR     Cache directory for thumbnails\n"
    "  --tmpdir DIR       Temporary files directory\n"
    "  --disable-opencl   Disable OpenCL GPU acceleration\n",
    progname);
}

// resolve path to the directory containing this binary
static char *_get_binary_dir(void)
{
  char self[PATH_MAX];

#ifdef __APPLE__
  uint32_t bufsize = sizeof(self);
  if(_NSGetExecutablePath(self, &bufsize) == 0)
  {
    char resolved[PATH_MAX];
    if(realpath(self, resolved))
      return g_path_get_dirname(resolved);
  }
#else
  ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if(len > 0)
  {
    self[len] = '\0';
    return g_path_get_dirname(self);
  }
#endif

  return NULL;
}

// spawn darktable-server and read SOCKET= from its stdout
// core_args is a NULL-terminated array of darktable options (passed after --core)
// returns PID on success, -1 on error; socket_path is filled in
static pid_t spawn_server(const char *server_bin, char **core_args, int core_argc,
                          char *socket_path, size_t path_size)
{
  int pipefd[2];
  if(pipe(pipefd) < 0)
  {
    perror("[webview] pipe");
    return -1;
  }

  pid_t pid = fork();
  if(pid < 0)
  {
    perror("[webview] fork");
    close(pipefd[0]);
    close(pipefd[1]);
    return -1;
  }

  if(pid == 0)
  {
    // child: redirect stdout to pipe, exec server
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);

    // build argv: server_bin [--core core_args...] NULL
    char **exec_argv = g_new0(char *, 1 + 1 + core_argc + 1);
    int n = 0;
    exec_argv[n++] = (char *)server_bin;
    if(core_argc > 0)
    {
      exec_argv[n++] = "--core";
      for(int i = 0; i < core_argc; i++)
        exec_argv[n++] = core_args[i];
    }
    exec_argv[n] = NULL;

    fprintf(stderr, "[webview] exec:");
    for(int i = 0; i < n; i++)
      fprintf(stderr, " %s", exec_argv[i]);
    fprintf(stderr, "\n");

    execvp(server_bin, exec_argv);
    perror("[webview] exec darktable-server");
    _exit(1);
  }

  // parent: read from pipe until SOCKET= line
  close(pipefd[1]);

  FILE *fp = fdopen(pipefd[0], "r");
  if(!fp)
  {
    perror("[webview] fdopen");
    close(pipefd[0]);
    kill(pid, SIGTERM);
    waitpid(pid, NULL, 0);
    return -1;
  }

  char line[1024];
  gboolean found = FALSE;
  while(fgets(line, sizeof(line), fp))
  {
    size_t len = strlen(line);
    if(len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';

    if(g_str_has_prefix(line, "SOCKET="))
    {
      g_strlcpy(socket_path, line + 7, path_size);
      found = TRUE;
      break;
    }
    // forward non-SOCKET lines to stderr
    fprintf(stderr, "[server] %s\n", line);
  }
  fclose(fp);

  if(!found)
  {
    fprintf(stderr, "[webview] server exited without printing SOCKET= line\n");
    kill(pid, SIGTERM);
    waitpid(pid, NULL, 0);
    return -1;
  }

  fprintf(stderr, "[webview] server started (pid=%d), socket: %s\n", pid, socket_path);
  return pid;
}

// called on main thread when server is ready
static void _on_server_ready(webview_t w, void *arg)
{
  _startup_data_t *data = arg;

  dt_webview_register_bindings(data->ctx);

  fprintf(stderr, "[webview] navigating to %s\n", data->frontend_url);
  webview_navigate(w, data->frontend_url);
}

// called on main thread when startup fails
static void _on_startup_error(webview_t w, void *arg)
{
  char *msg = arg;
  char *js = g_strdup_printf(
      "document.getElementById('splash-status').textContent='error: %s';"
      "document.getElementById('splash-status').style.color='#e55';",
      msg);
  webview_eval(w, js);
  g_free(js);
  g_free(msg);
}

// background thread: spawn server, connect, then dispatch to main thread
static void *_startup_thread(void *arg)
{
  _startup_data_t *data = arg;
  dt_webview_ctx_t *ctx = data->ctx;

  dt_splash_update(ctx->webview, "starting server...");

  fprintf(stderr, "[webview] starting server: %s\n", data->server_bin);
  ctx->server_pid = spawn_server(data->server_bin, data->core_args, data->core_argc,
                                 ctx->socket_path, sizeof(ctx->socket_path));
  if(ctx->server_pid < 0)
  {
    webview_dispatch(ctx->webview, _on_startup_error,
                     g_strdup("failed to start darktable-server"));
    return NULL;
  }

  dt_splash_update(ctx->webview, "connecting...");

  ctx->socket_fd = dt_ipc_connect(ctx->socket_path);
  if(ctx->socket_fd < 0)
  {
    kill(ctx->server_pid, SIGTERM);
    waitpid(ctx->server_pid, NULL, 0);
    ctx->server_pid = -1;
    webview_dispatch(ctx->webview, _on_startup_error,
                     g_strdup("failed to connect to server"));
    return NULL;
  }

  dt_splash_update(ctx->webview, "loading interface...");
  webview_dispatch(ctx->webview, _on_server_ready, data);
  return NULL;
}

int main(int argc, char *argv[])
{
  int dev_mode = 0;
  int server_mode = 0;
  const char *frontend_dir = NULL;
  const char *server_bin = NULL;

  // everything after --core is passed through to darktable-server
  char **core_args = NULL;
  int core_argc = 0;

  for(int i = 1; i < argc; i++)
  {
    if(!strcmp(argv[i], "--core"))
    {
      core_args = &argv[i + 1];
      core_argc = argc - (i + 1);
      break;
    }
    else if(!strcmp(argv[i], "--server"))
      server_mode = 1;
    else if(!strcmp(argv[i], "--dev"))
      dev_mode = 1;
    else if(!strcmp(argv[i], "--frontend-dir") && i + 1 < argc)
      frontend_dir = argv[++i];
    else if(!strcmp(argv[i], "--server-bin") && i + 1 < argc)
    {
      server_bin = argv[++i];
      server_mode = 1; // --server-bin implies --server
    }
    else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      usage(argv[0]);
      return 0;
    }
  }

  // resolve binary directory (for finding server binary and splash assets)
  char *binary_dir = _get_binary_dir();

  // resolve server binary path (only needed in server mode)
  if(server_mode)
  {
    if(!server_bin)
      server_bin = g_getenv("DT_SERVER_BIN");
    if(!server_bin && binary_dir)
    {
      char *candidate = g_build_filename(binary_dir, "darktable-server", NULL);
      if(g_file_test(candidate, G_FILE_TEST_IS_EXECUTABLE))
        server_bin = candidate;
      else
        g_free(candidate);
    }
    if(!server_bin)
      server_bin = "darktable-server";
  }

  // resolve frontend directory and URL
  if(!frontend_dir && !dev_mode)
  {
    frontend_dir = g_getenv("DT_FRONTEND_DIR");
    if(!frontend_dir)
      frontend_dir = "ui/dist";
  }

  char *frontend_url = NULL;
  if(dev_mode)
  {
    frontend_url = g_strdup("http://localhost:5173");
  }
  else
  {
    char *index_path = g_build_filename(frontend_dir, "index.html", NULL);
    if(!g_file_test(index_path, G_FILE_TEST_EXISTS))
    {
      fprintf(stderr, "ERROR: frontend not found at %s\n", index_path);
      fprintf(stderr, "Run 'cd ui && npm run build' first, or use --dev flag\n");
      g_free(index_path);
      g_free(binary_dir);
      return 1;
    }
    char resolved[PATH_MAX];
    if(!realpath(index_path, resolved))
    {
      fprintf(stderr, "ERROR: cannot resolve path %s: %s\n", index_path, strerror(errno));
      g_free(index_path);
      g_free(binary_dir);
      return 1;
    }
    g_free(index_path);
    frontend_url = g_strdup_printf("file://%s", resolved);
  }

  // init context
  dt_webview_ctx_t ctx;
  memset(&ctx, 0, sizeof(ctx));
  pthread_mutex_init(&ctx.ipc_mutex, NULL);
  pthread_mutex_init(&ctx.session_mutex, NULL);
  ctx.socket_fd = -1;

  // create webview first so we can show the splash during server startup
  ctx.webview = webview_create(1, NULL);
  if(!ctx.webview)
  {
    fprintf(stderr, "ERROR: failed to create webview\n");
    g_free(frontend_url);
    g_free(binary_dir);
    return 1;
  }

  webview_set_title(ctx.webview, "darktable");
  webview_set_size(ctx.webview, 1400, 900, WEBVIEW_HINT_NONE);

  // remove native titlebar, keep native window controls
  dt_titlebar_init(ctx.webview);

  // register window drag/zoom bindings early so splash is draggable
  dt_webview_register_window_bindings(&ctx);

  // show splash screen during startup
  dt_splash_show(ctx.webview, binary_dir);
  g_free(binary_dir);

  fprintf(stderr, "[webview] transport mode: %s\n", server_mode ? "IPC (server)" : "direct");

  pthread_t startup_thread = 0;
  _startup_data_t startup = { 0 };

  if(server_mode)
  {
    // IPC mode: spawn server and connect in a background thread
    startup = (_startup_data_t){
      .ctx = &ctx,
      .server_bin = server_bin,
      .core_args = core_args,
      .core_argc = core_argc,
      .frontend_url = frontend_url,
    };
    pthread_create(&startup_thread, NULL, _startup_thread, &startup);
  }
  else
  {
    // Direct mode: create in-process transport
    ctx.transport = dt_transport_direct_new();
    if(!ctx.transport)
    {
      fprintf(stderr, "ERROR: direct transport not yet implemented\n");
      fprintf(stderr, "Use --server flag to run with darktable-server\n");
      webview_destroy(ctx.webview);
      g_free(frontend_url);
      return 1;
    }
    fprintf(stderr, "[webview] direct transport ready\n");
    dt_webview_register_bindings(&ctx);
    webview_navigate(ctx.webview, frontend_url);
  }

  // run event loop (blocks until window is closed)
  webview_run(ctx.webview);

  // shutdown: kill server to unblock startup thread if it's still waiting
  if(ctx.server_pid > 0)
    kill(ctx.server_pid, SIGTERM);

  if(server_mode)
    pthread_join(startup_thread, NULL);

  // cleanup
  fprintf(stderr, "[webview] shutting down...\n");
  dt_binding_pool_shutdown();
  webview_destroy(ctx.webview);

  // Send graceful shutdown to server (before closing IPC) so it saves config
  if(ctx.server_pid > 0 && ctx.transport)
  {
    char *error = NULL;
    char *resp = dt_transport_call(ctx.transport, "system.shutdown", "{}", &error);
    g_free(resp);
    g_free(error);
  }

  // Shut down frame server and IPC reader thread before closing the socket
  if(ctx.frame_server)
    dt_frame_server_stop(ctx.frame_server);
  if(ctx.ipc_ctx)
    dt_ipc_context_free(ctx.ipc_ctx);

  if(ctx.transport)
  {
    dt_transport_destroy(ctx.transport);
    ctx.transport = NULL;
  }

  if(ctx.socket_fd >= 0)
    close(ctx.socket_fd);

  // close any open SHM handles
  for(int i = 0; i < DT_WEBVIEW_MAX_SESSIONS; i++)
  {
    if(ctx.sessions[i].active)
    {
      for(int b = 0; b < 2; b++)
      {
        if(ctx.sessions[i].shm_ptr[b])
          munmap(ctx.sessions[i].shm_ptr[b], ctx.sessions[i].shm_size[b]);
      }
    }
  }

  // wait for server to exit cleanly, force kill if needed
  if(ctx.server_pid > 0)
  {
    int status;
    for(int i = 0; i < 30; i++)
    {
      pid_t ret = waitpid(ctx.server_pid, &status, WNOHANG);
      if(ret != 0) goto server_done;
      usleep(100000); // 100ms
    }
    fprintf(stderr, "[webview] server did not exit, sending SIGTERM\n");
    kill(ctx.server_pid, SIGTERM);
    waitpid(ctx.server_pid, &status, 0);
server_done:
    fprintf(stderr, "[webview] server stopped\n");
  }

  pthread_mutex_destroy(&ctx.ipc_mutex);
  pthread_mutex_destroy(&ctx.session_mutex);
  g_free(frontend_url);

  fprintf(stderr, "[webview] done\n");
  return 0;
}
