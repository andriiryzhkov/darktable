#pragma once

#include <webview/api.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/types.h>

#define DT_WEBVIEW_MAX_SESSIONS 4

typedef struct dt_webview_shm_t
{
  char session_id[64];
  char shm_names[2][64];
  void *shm_ptr[2];
  size_t shm_size[2];
  uint32_t width;
  uint32_t height;
  int active;
} dt_webview_shm_t;

typedef struct dt_webview_ctx_t
{
  webview_t webview;
  int socket_fd;
  pid_t server_pid;
  char socket_path[PATH_MAX];
  pthread_mutex_t ipc_mutex;
  dt_webview_shm_t sessions[DT_WEBVIEW_MAX_SESSIONS];
  pthread_mutex_t session_mutex;
} dt_webview_ctx_t;

// Register all JS bindings on the webview
void dt_webview_register_bindings(dt_webview_ctx_t *ctx);
