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

#include "common/darktable.h"
#include "common/file_location.h"
#include "server/server.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __APPLE__
#include "osx/osx.h"
#endif

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

static void usage(const char *progname)
{
  fprintf(stderr,
    "darktable-server %s\n"
    "Copyright (C) 2025 darktable developers.\n\n"
    "Headless server for the Tauri UI frontend.\n\n"
    "Usage:\n"
    "  %s [OPTIONS] [--core DARKTABLE_OPTIONS]\n\n"
    "Options:\n"
    "  --socket PATH    Unix socket path (default: auto-generated)\n"
    "  -h, --help       Show this help\n"
    "  -v, --version    Show version\n\n"
    "darktable options (after --core):\n"
    "  --configdir DIR  Use DIR for config/library (useful for testing)\n"
    "  --library FILE   Use specific library.db file\n"
    "  --disable-opencl Disable OpenCL GPU acceleration\n",
    darktable_package_version, progname);
}

int main(int argc, char *argv[])
{
#ifdef __APPLE__
  dt_osx_prepare_environment();
#endif

  // Parse server-specific arguments (before --core)
  char *socket_path = NULL;
  int core_argc_start = argc; // index of first --core arg

  for(int k = 1; k < argc; k++)
  {
    if(!strcmp(argv[k], "--socket") && k + 1 < argc)
    {
      socket_path = argv[++k];
    }
    else if(!strcmp(argv[k], "--help") || !strcmp(argv[k], "-h"))
    {
      usage(argv[0]);
      exit(0);
    }
    else if(!strcmp(argv[k], "--version") || !strcmp(argv[k], "-v"))
    {
      printf("darktable-server %s\n", darktable_package_version);
      exit(0);
    }
    else if(!strcmp(argv[k], "--core"))
    {
      core_argc_start = k + 1;
      break;
    }
  }

  // Build args for dt_init: program name + everything after --core
  int dt_argc = 1 + (argc - core_argc_start);
  char **dt_argv = malloc(sizeof(char *) * (dt_argc + 1));
  dt_argv[0] = "darktable-server";
  for(int k = core_argc_start; k < argc; k++)
    dt_argv[1 + k - core_argc_start] = argv[k];
  dt_argv[dt_argc] = NULL;

  // Initialize darktable headlessly with REAL library database.
  // init_gui=FALSE: no GTK, no GUI widgets
  // load_data=TRUE: load presets, styles, etc.
  fprintf(stderr, "[server] initializing darktable (headless)...\n");
  if(dt_init(dt_argc, dt_argv, FALSE, TRUE, NULL))
  {
    fprintf(stderr, "ERROR: failed to initialize darktable\n");
    free(dt_argv);
    exit(1);
  }
  fprintf(stderr, "[server] darktable initialized successfully\n");

  // Auto-generate socket path if not specified
  char auto_socket[PATH_MAX];
  if(!socket_path)
  {
    const char *runtime_dir = g_get_user_runtime_dir();
    snprintf(auto_socket, sizeof(auto_socket),
             "%s/darktable-server.%d.sock", runtime_dir, getpid());
    socket_path = auto_socket;
  }

  // Create server (binds + listens on socket)
  dt_server_t *server = dt_server_init(socket_path);
  if(!server)
  {
    fprintf(stderr, "ERROR: failed to start server on %s\n", socket_path);
    dt_cleanup();
    free(dt_argv);
    exit(1);
  }

  // Print socket path AFTER the socket is listening so the client can connect immediately
  fprintf(stdout, "SOCKET=%s\n", socket_path);
  fflush(stdout);

  fprintf(stderr, "[server] running (pid=%d)\n", getpid());
  dt_server_run(server); // blocks until shutdown

  fprintf(stderr, "[server] shutting down...\n");
  dt_server_cleanup(server);
  dt_cleanup();
  free(dt_argv);

  fprintf(stderr, "[server] done\n");
  return 0;
}
