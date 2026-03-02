#include "webview_bindings.h"
#include "ipc_client.h"

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

static void usage(const char *progname)
{
  fprintf(stderr,
    "darktable-webview — webview UI for darktable\n\n"
    "Usage:\n"
    "  %s [OPTIONS]\n\n"
    "Options:\n"
    "  --dev              Connect to Vite dev server at http://localhost:5173\n"
    "  --frontend-dir DIR Load production build from DIR (default: ui/dist)\n"
    "  --server-bin PATH  Path to darktable-server binary\n"
    "  --configdir DIR    Config directory for darktable-server\n"
    "  -h, --help         Show this help\n",
    progname);
}

// Spawn darktable-server and read SOCKET= from its stdout.
// Returns PID on success, -1 on error. socket_path is filled in.
static pid_t spawn_server(const char *server_bin, const char *configdir, char *socket_path, size_t path_size)
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
    // Child: redirect stdout to pipe, exec server
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[1]);

    execlp(server_bin, server_bin, "--core", "--configdir", configdir, NULL);
    perror("[webview] exec darktable-server");
    _exit(1);
  }

  // Parent: read from pipe until SOCKET= line
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
    // Strip trailing newline
    size_t len = strlen(line);
    if(len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';

    if(g_str_has_prefix(line, "SOCKET="))
    {
      g_strlcpy(socket_path, line + 7, path_size);
      found = TRUE;
      break;
    }
    // Forward non-SOCKET lines to stderr
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

int main(int argc, char *argv[])
{
  int dev_mode = 0;
  const char *frontend_dir = NULL;
  const char *server_bin = NULL;
  const char *configdir = NULL;

  for(int i = 1; i < argc; i++)
  {
    if(!strcmp(argv[i], "--dev"))
      dev_mode = 1;
    else if(!strcmp(argv[i], "--frontend-dir") && i + 1 < argc)
      frontend_dir = argv[++i];
    else if(!strcmp(argv[i], "--server-bin") && i + 1 < argc)
      server_bin = argv[++i];
    else if(!strcmp(argv[i], "--configdir") && i + 1 < argc)
      configdir = argv[++i];
    else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      usage(argv[0]);
      return 0;
    }
  }

  // Defaults
  if(!server_bin)
    server_bin = g_getenv("DT_SERVER_BIN");
  if(!server_bin)
  {
    // Look relative to this binary for darktable-server
    char self[PATH_MAX];
    gboolean found_self = FALSE;

#ifdef __APPLE__
    uint32_t bufsize = sizeof(self);
    if(_NSGetExecutablePath(self, &bufsize) == 0)
    {
      // Resolve symlinks to get canonical path
      char resolved[PATH_MAX];
      if(realpath(self, resolved))
      {
        g_strlcpy(self, resolved, sizeof(self));
        found_self = TRUE;
      }
    }
#else
    ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if(len > 0)
    {
      self[len] = '\0';
      found_self = TRUE;
    }
#endif

    if(found_self)
    {
      char *dir = g_path_get_dirname(self);
      char *candidate = g_build_filename(dir, "darktable-server", NULL);
      if(g_file_test(candidate, G_FILE_TEST_IS_EXECUTABLE))
        server_bin = candidate;
      else
        server_bin = "darktable-server";
      g_free(dir);
      // Note: candidate may leak if used, but that's fine for startup
    }
    else
    {
      server_bin = "darktable-server";
    }
  }

  if(!configdir)
    configdir = g_getenv("DT_CONFIGDIR");
  if(!configdir)
  {
    const char *home = g_get_home_dir();
    configdir = g_build_filename(home, ".config", "darktable-webview-test", NULL);
  }

  if(!frontend_dir && !dev_mode)
  {
    // Default: look for ui/dist relative to binary
    frontend_dir = g_getenv("DT_FRONTEND_DIR");
    if(!frontend_dir)
      frontend_dir = "ui/dist";
  }

  // Spawn server
  dt_webview_ctx_t ctx;
  memset(&ctx, 0, sizeof(ctx));
  pthread_mutex_init(&ctx.ipc_mutex, NULL);
  pthread_mutex_init(&ctx.session_mutex, NULL);
  ctx.socket_fd = -1;

  fprintf(stderr, "[webview] starting server: %s\n", server_bin);
  fprintf(stderr, "[webview] configdir: %s\n", configdir);

  ctx.server_pid = spawn_server(server_bin, configdir, ctx.socket_path, sizeof(ctx.socket_path));
  if(ctx.server_pid < 0)
  {
    fprintf(stderr, "ERROR: failed to start darktable-server\n");
    return 1;
  }

  // Connect IPC
  ctx.socket_fd = dt_ipc_connect(ctx.socket_path);
  if(ctx.socket_fd < 0)
  {
    fprintf(stderr, "ERROR: failed to connect to server\n");
    kill(ctx.server_pid, SIGTERM);
    waitpid(ctx.server_pid, NULL, 0);
    return 1;
  }

  // Create webview
  ctx.webview = webview_create(1, NULL);
  if(!ctx.webview)
  {
    fprintf(stderr, "ERROR: failed to create webview\n");
    close(ctx.socket_fd);
    kill(ctx.server_pid, SIGTERM);
    waitpid(ctx.server_pid, NULL, 0);
    return 1;
  }

  webview_set_title(ctx.webview, "darktable");
  webview_set_size(ctx.webview, 1400, 900, WEBVIEW_HINT_NONE);

  // Register JS bindings
  dt_webview_register_bindings(&ctx);

  // Navigate to frontend
  if(dev_mode)
  {
    fprintf(stderr, "[webview] dev mode: navigating to http://localhost:5173\n");
    webview_navigate(ctx.webview, "http://localhost:5173");
  }
  else
  {
    char *index_path = g_build_filename(frontend_dir, "index.html", NULL);
    if(!g_file_test(index_path, G_FILE_TEST_EXISTS))
    {
      fprintf(stderr, "ERROR: frontend not found at %s\n", index_path);
      fprintf(stderr, "Run 'cd ui && npm run build' first, or use --dev flag\n");
      g_free(index_path);
      webview_destroy(ctx.webview);
      close(ctx.socket_fd);
      kill(ctx.server_pid, SIGTERM);
      waitpid(ctx.server_pid, NULL, 0);
      return 1;
    }
    char *url = g_strdup_printf("file://%s", index_path);
    fprintf(stderr, "[webview] navigating to %s\n", url);
    webview_navigate(ctx.webview, url);
    g_free(url);
    g_free(index_path);
  }

  // Run event loop (blocks until window is closed)
  webview_run(ctx.webview);

  // Cleanup
  fprintf(stderr, "[webview] shutting down...\n");
  webview_destroy(ctx.webview);

  if(ctx.socket_fd >= 0)
    close(ctx.socket_fd);

  // Close any open SHM handles
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

  // Stop server
  if(ctx.server_pid > 0)
  {
    kill(ctx.server_pid, SIGTERM);
    int status;
    waitpid(ctx.server_pid, &status, 0);
    fprintf(stderr, "[webview] server stopped\n");
  }

  pthread_mutex_destroy(&ctx.ipc_mutex);
  pthread_mutex_destroy(&ctx.session_mutex);

  fprintf(stderr, "[webview] done\n");
  return 0;
}
