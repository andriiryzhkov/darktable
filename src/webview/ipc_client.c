#include "ipc_client.h"
#include "server/server_protocol.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static volatile uint64_t _next_id = 1;

int dt_ipc_connect(const char *socket_path)
{
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if(fd < 0)
  {
    fprintf(stderr, "[webview] socket() failed: %s\n", strerror(errno));
    return -1;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  g_strlcpy(addr.sun_path, socket_path, sizeof(addr.sun_path));

  if(connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
  {
    fprintf(stderr, "[webview] connect(%s) failed: %s\n", socket_path, strerror(errno));
    close(fd);
    return -1;
  }

  fprintf(stderr, "[webview] connected to server: %s\n", socket_path);
  return fd;
}

char *dt_ipc_request(int fd, pthread_mutex_t *mutex,
                     const char *method, const char *params_json,
                     char **out_error)
{
  if(out_error) *out_error = NULL;

  // Build request JSON: {"id":"req-N","method":"...","params":{...}}
  uint64_t id = __atomic_fetch_add(&_next_id, 1, __ATOMIC_RELAXED);
  char id_str[32];
  snprintf(id_str, sizeof(id_str), "req-%" PRIu64, id);

  char *request;
  if(params_json && params_json[0])
    request = g_strdup_printf("{\"id\":\"%s\",\"method\":\"%s\",\"params\":%s}",
                              id_str, method, params_json);
  else
    request = g_strdup_printf("{\"id\":\"%s\",\"method\":\"%s\",\"params\":{}}",
                              id_str, method);

  char *result_json = NULL;

  pthread_mutex_lock(mutex);

  // Send request frame
  if(!dt_server_write_frame(fd, request, strlen(request)))
  {
    if(out_error) *out_error = g_strdup("failed to write request frame");
    g_free(request);
    pthread_mutex_unlock(mutex);
    return NULL;
  }
  g_free(request);

  // Read response(s), skipping events (id=null)
  while(TRUE)
  {
    char *frame = NULL;
    size_t frame_len = 0;
    if(!dt_server_read_frame(fd, &frame, &frame_len))
    {
      if(out_error) *out_error = g_strdup("failed to read response frame");
      pthread_mutex_unlock(mutex);
      return NULL;
    }

    // Parse JSON
    JsonParser *parser = json_parser_new();
    if(!json_parser_load_from_data(parser, frame, frame_len, NULL))
    {
      if(out_error) *out_error = g_strdup("failed to parse response JSON");
      g_free(frame);
      g_object_unref(parser);
      pthread_mutex_unlock(mutex);
      return NULL;
    }
    g_free(frame);

    JsonNode *root = json_parser_get_root(parser);
    JsonObject *obj = json_node_get_object(root);

    // Skip events (id is null or missing, has "event" field)
    if(json_object_has_member(obj, "event"))
    {
      fprintf(stderr, "[webview] skipping event: %s\n",
              json_object_get_string_member(obj, "event"));
      g_object_unref(parser);
      continue;
    }

    // Check for error
    if(json_object_has_member(obj, "error") && !json_object_get_null_member(obj, "error"))
    {
      JsonObject *err = json_object_get_object_member(obj, "error");
      const char *msg = json_object_get_string_member(err, "message");
      if(out_error) *out_error = g_strdup(msg ? msg : "unknown server error");
      g_object_unref(parser);
      pthread_mutex_unlock(mutex);
      return NULL;
    }

    // Extract result
    if(json_object_has_member(obj, "result") && !json_object_get_null_member(obj, "result"))
    {
      JsonNode *result_node = json_object_get_member(obj, "result");
      JsonGenerator *gen = json_generator_new();
      json_generator_set_root(gen, result_node);
      result_json = json_generator_to_data(gen, NULL);
      g_object_unref(gen);
    }
    else
    {
      result_json = g_strdup("null");
    }

    g_object_unref(parser);
    break;
  }

  pthread_mutex_unlock(mutex);
  return result_json;
}
