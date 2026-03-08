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

/*
 * Unit tests for the transport vtable (transport.h).
 *
 * Since transport implementations (dt_transport_ipc_new, dt_transport_direct_new)
 * are not yet implemented, tests use inline null and loopback transports to
 * exercise the vtable interface, convenience macros, and lifecycle.
 *
 * The loopback transport uses a socketpair + echo responder thread with the
 * real frame I/O protocol, validating the IPC serialization path.
 */

#include <pthread.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmocka.h>

#include "server/server_protocol.h"
#include "webview/transport.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

/* ── Null transport ───────────────────────────────────────────── */

static char *_null_call(dt_webview_transport_t *self,
                        const char *method,
                        const char *params_json,
                        char **error)
{
  (void)self; (void)method; (void)params_json; (void)error;
  return g_strdup("{\"status\":\"ok\"}");
}

static gboolean _null_get_frame(dt_webview_transport_t *self,
                                const char *session_id,
                                dt_transport_frame_t *out_frame)
{
  (void)self; (void)session_id; (void)out_frame;
  return FALSE;
}

static void _null_set_event_cb(dt_webview_transport_t *self,
                               dt_transport_event_cb callback,
                               void *user_data)
{
  (void)self; (void)callback; (void)user_data;
}

static void _null_destroy(dt_webview_transport_t *self)
{
  g_free(self);
}

static dt_webview_transport_t *_null_transport_new(void)
{
  dt_webview_transport_t *t = g_new0(dt_webview_transport_t, 1);
  t->call = _null_call;
  t->get_preview_frame = _null_get_frame;
  t->set_event_callback = _null_set_event_cb;
  t->destroy = _null_destroy;
  return t;
}

/* ── Error transport (always fails) ───────────────────────────── */

static char *_error_call(dt_webview_transport_t *self,
                         const char *method,
                         const char *params_json,
                         char **error)
{
  (void)self; (void)method; (void)params_json;
  if(error) *error = g_strdup("Transport error: connection refused");
  return NULL;
}

static dt_webview_transport_t *_error_transport_new(void)
{
  dt_webview_transport_t *t = g_new0(dt_webview_transport_t, 1);
  t->call = _error_call;
  t->get_preview_frame = _null_get_frame;
  t->set_event_callback = _null_set_event_cb;
  t->destroy = _null_destroy;
  return t;
}

/* ── Loopback transport ───────────────────────────────────────── */

typedef struct loopback_data_t
{
  int client_fd;
  int server_fd;
  pthread_t thread;
  volatile gboolean running;
} loopback_data_t;

static void *_echo_responder(void *arg)
{
  loopback_data_t *lb = arg;
  char *buf = NULL;
  size_t len = 0;

  while(lb->running)
  {
    if(!dt_server_read_frame(lb->server_fd, &buf, &len))
      break;

    dt_server_request_t *req = dt_server_parse_request(buf, len);
    g_free(buf);
    buf = NULL;

    if(!req) continue;

    JsonBuilder *b = json_builder_new();
    json_builder_begin_object(b);
    json_builder_set_member_name(b, "status");
    json_builder_add_string_value(b, "ok");
    json_builder_set_member_name(b, "method");
    json_builder_add_string_value(b, req->method);
    json_builder_end_object(b);
    JsonNode *result_node = json_builder_get_root(b);

    char *resp = dt_server_make_response(req->id, result_node);
    dt_server_write_frame(lb->server_fd, resp, strlen(resp));

    json_node_unref(result_node);
    g_object_unref(b);
    g_free(resp);
    dt_server_free_request(req);
  }

  return NULL;
}

static char *_loopback_call(dt_webview_transport_t *self,
                            const char *method,
                            const char *params_json,
                            char **error)
{
  loopback_data_t *lb = self->data;

  static int req_id = 0;
  char *request;
  if(params_json)
    request = g_strdup_printf("{\"id\":\"%d\",\"method\":\"%s\",\"params\":%s}",
                              ++req_id, method, params_json);
  else
    request = g_strdup_printf("{\"id\":\"%d\",\"method\":\"%s\"}", ++req_id, method);

  if(!dt_server_write_frame(lb->client_fd, request, strlen(request)))
  {
    g_free(request);
    if(error) *error = g_strdup("write failed");
    return NULL;
  }
  g_free(request);

  char *resp = NULL;
  size_t resp_len = 0;
  if(!dt_server_read_frame(lb->client_fd, &resp, &resp_len))
  {
    if(error) *error = g_strdup("read failed");
    return NULL;
  }

  return resp;
}

static void _loopback_destroy(dt_webview_transport_t *self)
{
  loopback_data_t *lb = self->data;
  lb->running = FALSE;
  shutdown(lb->server_fd, SHUT_RDWR);
  close(lb->server_fd);
  pthread_join(lb->thread, NULL);
  close(lb->client_fd);
  g_free(lb);
  g_free(self);
}

static dt_webview_transport_t *_loopback_transport_new(void)
{
  int fds[2];
  if(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) < 0)
    return NULL;

  loopback_data_t *lb = g_new0(loopback_data_t, 1);
  lb->client_fd = fds[0];
  lb->server_fd = fds[1];
  lb->running = TRUE;

  if(pthread_create(&lb->thread, NULL, _echo_responder, lb) != 0)
  {
    close(fds[0]);
    close(fds[1]);
    g_free(lb);
    return NULL;
  }

  dt_webview_transport_t *t = g_new0(dt_webview_transport_t, 1);
  t->data = lb;
  t->call = _loopback_call;
  t->get_preview_frame = _null_get_frame;
  t->set_event_callback = _null_set_event_cb;
  t->destroy = _loopback_destroy;
  return t;
}

/* ── Null transport tests ─────────────────────────────────────── */

static void test_null_call_returns_json(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _null_transport_new();

  char *result = dt_transport_call(t, "system.ping", NULL, NULL);
  assert_non_null(result);
  assert_string_equal(result, "{\"status\":\"ok\"}");

  g_free(result);
  dt_transport_destroy(t);
}

static void test_null_call_with_params(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _null_transport_new();

  char *result = dt_transport_call(t, "develop.set_params",
    "{\"op\":\"exposure\",\"params\":{\"exposure\":1.5}}", NULL);
  assert_non_null(result);

  g_free(result);
  dt_transport_destroy(t);
}

static void test_null_get_preview_frame_returns_false(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _null_transport_new();

  dt_transport_frame_t frame = { 0 };
  assert_false(dt_transport_get_preview_frame(t, "session1", &frame));

  dt_transport_destroy(t);
}

static void test_null_event_callback_noop(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _null_transport_new();

  /* Should not crash */
  dt_transport_set_event_callback(t, NULL, NULL);

  dt_transport_destroy(t);
}

/* ── Error transport tests ────────────────────────────────────── */

static void test_error_call_returns_null(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _error_transport_new();

  char *error = NULL;
  char *result = dt_transport_call(t, "system.ping", NULL, &error);
  assert_null(result);
  assert_non_null(error);
  assert_true(strstr(error, "connection refused") != NULL);

  g_free(error);
  dt_transport_destroy(t);
}

static void test_error_call_null_error_ptr(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _error_transport_new();

  /* Should not crash even with NULL error pointer */
  char *result = dt_transport_call(t, "system.ping", NULL, NULL);
  assert_null(result);

  dt_transport_destroy(t);
}

/* ── Loopback transport tests ─────────────────────────────────── */

static void test_loopback_call_roundtrip(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _loopback_transport_new();
  assert_non_null(t);

  char *result = dt_transport_call(t, "system.ping", NULL, NULL);
  assert_non_null(result);

  /* Verify the response is valid JSON with our echo structure */
  JsonParser *p = json_parser_new();
  assert_true(json_parser_load_from_data(p, result, -1, NULL));
  JsonObject *obj = json_node_get_object(json_parser_get_root(p));
  assert_true(json_object_has_member(obj, "result"));

  g_object_unref(p);
  g_free(result);
  dt_transport_destroy(t);
}

static void test_loopback_echoes_method(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _loopback_transport_new();
  assert_non_null(t);

  char *result = dt_transport_call(t, "develop.set_params",
    "{\"op\":\"exposure\"}", NULL);
  assert_non_null(result);

  /* Echo responder includes the method name in the result */
  JsonParser *p = json_parser_new();
  assert_true(json_parser_load_from_data(p, result, -1, NULL));
  JsonObject *root = json_node_get_object(json_parser_get_root(p));
  JsonObject *res = json_object_get_object_member(root, "result");
  assert_string_equal(json_object_get_string_member(res, "method"), "develop.set_params");

  g_object_unref(p);
  g_free(result);
  dt_transport_destroy(t);
}

static void test_loopback_multiple_calls(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _loopback_transport_new();
  assert_non_null(t);

  const char *methods[] = { "system.ping", "catalog.query", "develop.open" };
  for(int i = 0; i < 3; i++)
  {
    char *result = dt_transport_call(t, methods[i], NULL, NULL);
    assert_non_null(result);

    JsonParser *p = json_parser_new();
    assert_true(json_parser_load_from_data(p, result, -1, NULL));
    JsonObject *root = json_node_get_object(json_parser_get_root(p));
    JsonObject *res = json_object_get_object_member(root, "result");
    assert_string_equal(json_object_get_string_member(res, "method"), methods[i]);

    g_object_unref(p);
    g_free(result);
  }

  dt_transport_destroy(t);
}

static void test_loopback_with_complex_params(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _loopback_transport_new();
  assert_non_null(t);

  const char *params =
    "{\"session_id\":\"dev001\",\"op\":\"exposure\","
    "\"multi_instance\":0,\"params\":{\"exposure\":1.5,\"black\":0.01},"
    "\"preview_only\":true}";

  char *result = dt_transport_call(t, "develop.set_params", params, NULL);
  assert_non_null(result);

  g_free(result);
  dt_transport_destroy(t);
}

/* ── Lifecycle tests ──────────────────────────────────────────── */

static void test_create_destroy_null(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _null_transport_new();
  assert_non_null(t);
  assert_non_null(t->call);
  assert_non_null(t->get_preview_frame);
  assert_non_null(t->set_event_callback);
  assert_non_null(t->destroy);
  dt_transport_destroy(t);
}

static void test_create_destroy_loopback(void **state)
{
  (void)state;
  dt_webview_transport_t *t = _loopback_transport_new();
  assert_non_null(t);
  dt_transport_destroy(t);
}

static void test_rapid_create_destroy(void **state)
{
  (void)state;
  /* Create and destroy 10 loopback transports to check for leaks/hangs */
  for(int i = 0; i < 10; i++)
  {
    dt_webview_transport_t *t = _loopback_transport_new();
    assert_non_null(t);
    char *r = dt_transport_call(t, "system.ping", NULL, NULL);
    g_free(r);
    dt_transport_destroy(t);
  }
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  const struct CMUnitTest tests[] = {
    /* Null transport */
    cmocka_unit_test(test_null_call_returns_json),
    cmocka_unit_test(test_null_call_with_params),
    cmocka_unit_test(test_null_get_preview_frame_returns_false),
    cmocka_unit_test(test_null_event_callback_noop),
    /* Error transport */
    cmocka_unit_test(test_error_call_returns_null),
    cmocka_unit_test(test_error_call_null_error_ptr),
    /* Loopback transport (IPC serialization) */
    cmocka_unit_test(test_loopback_call_roundtrip),
    cmocka_unit_test(test_loopback_echoes_method),
    cmocka_unit_test(test_loopback_multiple_calls),
    cmocka_unit_test(test_loopback_with_complex_params),
    /* Lifecycle */
    cmocka_unit_test(test_create_destroy_null),
    cmocka_unit_test(test_create_destroy_loopback),
    cmocka_unit_test(test_rapid_create_destroy),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
