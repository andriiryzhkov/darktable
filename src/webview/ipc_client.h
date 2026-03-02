#pragma once

#include <glib.h>
#include <json-glib/json-glib.h>
#include <pthread.h>

// Connect to darktable-server Unix socket. Returns fd >= 0 on success, -1 on error.
int dt_ipc_connect(const char *socket_path);

// Send a JSON-RPC request and wait for the response.
// Thread-safe when protected by mutex.
// Returns the "result" field as a JSON string (caller must g_free), or NULL on error.
// On error, *out_error is set to a newly-allocated error string (caller must g_free).
char *dt_ipc_request(int fd, pthread_mutex_t *mutex,
                     const char *method, const char *params_json,
                     char **out_error);
