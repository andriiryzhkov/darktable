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
 * Unit tests for the JSON-RPC protocol layer (server_protocol.c).
 * Tests frame I/O, request parsing, response/error/event building.
 */

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmocka.h>

#include "server/server_protocol.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

/* ── Frame encode/decode tests ────────────────────────────────── */

static void test_frame_roundtrip(void **state)
{
  (void)state;
  int fds[2];
  assert_int_equal(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  const char *msg = "{\"id\":\"1\",\"method\":\"system.ping\"}";
  size_t msg_len = strlen(msg);

  /* Write frame on one end */
  assert_true(dt_server_write_frame(fds[0], msg, msg_len));

  /* Read frame on the other end */
  char *out = NULL;
  size_t out_len = 0;
  assert_true(dt_server_read_frame(fds[1], &out, &out_len));

  assert_int_equal(out_len, msg_len);
  assert_string_equal(out, msg);

  g_free(out);
  close(fds[0]);
  close(fds[1]);
}

static void test_frame_roundtrip_large(void **state)
{
  (void)state;
  int fds[2];
  assert_int_equal(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  /* Use 4 KB — must fit in socketpair buffer to avoid deadlock
   * (single-threaded: write then read on same thread) */
  size_t len = 4 * 1024;
  char *big = g_malloc(len + 1);
  memset(big, 'X', len);
  big[len] = '\0';

  assert_true(dt_server_write_frame(fds[0], big, len));

  char *out = NULL;
  size_t out_len = 0;
  assert_true(dt_server_read_frame(fds[1], &out, &out_len));

  assert_int_equal(out_len, len);
  assert_memory_equal(out, big, len);

  g_free(out);
  g_free(big);
  close(fds[0]);
  close(fds[1]);
}

static void test_frame_multiple(void **state)
{
  (void)state;
  int fds[2];
  assert_int_equal(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  const char *msgs[] = { "first", "second", "third" };
  int n = 3;

  /* Write all frames */
  for(int i = 0; i < n; i++)
    assert_true(dt_server_write_frame(fds[0], msgs[i], strlen(msgs[i])));

  /* Read them back in order */
  for(int i = 0; i < n; i++)
  {
    char *out = NULL;
    size_t out_len = 0;
    assert_true(dt_server_read_frame(fds[1], &out, &out_len));
    assert_string_equal(out, msgs[i]);
    g_free(out);
  }

  close(fds[0]);
  close(fds[1]);
}

static void test_frame_read_closed_fd(void **state)
{
  (void)state;
  int fds[2];
  assert_int_equal(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  /* Close the write end immediately */
  close(fds[0]);

  char *out = NULL;
  size_t out_len = 0;
  assert_false(dt_server_read_frame(fds[1], &out, &out_len));
  assert_null(out);

  close(fds[1]);
}

/* ── Frame size validation ────────────────────────────────────── */

static void test_frame_zero_length_rejected(void **state)
{
  (void)state;
  int fds[2];
  assert_int_equal(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  /* Write a zero-length frame header manually */
  uint8_t zero_hdr[4] = { 0, 0, 0, 0 };
  assert_int_equal(write(fds[0], zero_hdr, 4), 4);
  close(fds[0]);

  char *out = NULL;
  size_t out_len = 0;
  assert_false(dt_server_read_frame(fds[1], &out, &out_len));
  assert_null(out);

  close(fds[1]);
}

static void test_frame_oversized_rejected(void **state)
{
  (void)state;
  int fds[2];
  assert_int_equal(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

  /* Write a header claiming > DT_SERVER_MAX_MESSAGE_SIZE (16 MB) */
  uint32_t too_big = DT_SERVER_MAX_MESSAGE_SIZE + 1;
  uint8_t hdr[4];
  hdr[0] = (too_big >> 24) & 0xFF;
  hdr[1] = (too_big >> 16) & 0xFF;
  hdr[2] = (too_big >> 8) & 0xFF;
  hdr[3] = too_big & 0xFF;
  assert_int_equal(write(fds[0], hdr, 4), 4);
  close(fds[0]);

  char *out = NULL;
  size_t out_len = 0;
  assert_false(dt_server_read_frame(fds[1], &out, &out_len));
  assert_null(out);

  close(fds[1]);
}

/* ── JSON-RPC request parsing ─────────────────────────────────── */

static void test_parse_valid_request(void **state)
{
  (void)state;
  const char *json = "{\"id\":\"42\",\"method\":\"system.ping\",\"params\":{}}";
  dt_server_request_t *req = dt_server_parse_request(json, strlen(json));

  assert_non_null(req);
  assert_string_equal(req->id, "42");
  assert_string_equal(req->method, "system.ping");
  assert_non_null(req->params);

  dt_server_free_request(req);
}

static void test_parse_request_no_params(void **state)
{
  (void)state;
  const char *json = "{\"id\":\"1\",\"method\":\"system.ping\"}";
  dt_server_request_t *req = dt_server_parse_request(json, strlen(json));

  assert_non_null(req);
  assert_string_equal(req->method, "system.ping");
  assert_null(req->params);

  dt_server_free_request(req);
}

static void test_parse_request_missing_method(void **state)
{
  (void)state;
  const char *json = "{\"id\":\"1\",\"params\":{}}";
  dt_server_request_t *req = dt_server_parse_request(json, strlen(json));
  assert_null(req);
}

static void test_parse_request_invalid_json(void **state)
{
  (void)state;
  const char *json = "not json at all";
  dt_server_request_t *req = dt_server_parse_request(json, strlen(json));
  assert_null(req);
}

static void test_parse_request_empty_string(void **state)
{
  (void)state;
  dt_server_request_t *req = dt_server_parse_request("", 0);
  assert_null(req);
}

static void test_parse_request_array_not_object(void **state)
{
  (void)state;
  const char *json = "[1, 2, 3]";
  dt_server_request_t *req = dt_server_parse_request(json, strlen(json));
  assert_null(req);
}

static void test_parse_request_no_id(void **state)
{
  (void)state;
  /* Missing id should still parse (id defaults to "") */
  const char *json = "{\"method\":\"system.ping\"}";
  dt_server_request_t *req = dt_server_parse_request(json, strlen(json));

  assert_non_null(req);
  assert_string_equal(req->id, "");
  assert_string_equal(req->method, "system.ping");

  dt_server_free_request(req);
}

/* ── Response / error / event building ────────────────────────── */

static void test_make_response(void **state)
{
  (void)state;
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "status");
  json_builder_add_string_value(b, "ok");
  json_builder_end_object(b);
  JsonNode *result = json_builder_get_root(b);

  char *resp = dt_server_make_response("42", result);
  assert_non_null(resp);

  /* Parse and verify structure */
  JsonParser *p = json_parser_new();
  assert_true(json_parser_load_from_data(p, resp, -1, NULL));
  JsonObject *obj = json_node_get_object(json_parser_get_root(p));

  assert_string_equal(json_object_get_string_member(obj, "id"), "42");
  assert_true(json_object_has_member(obj, "result"));
  assert_true(json_object_get_null_member(obj, "error"));

  JsonObject *res_obj = json_object_get_object_member(obj, "result");
  assert_string_equal(json_object_get_string_member(res_obj, "status"), "ok");

  g_object_unref(p);
  g_free(resp);
  json_node_unref(result);
  g_object_unref(b);
}

static void test_make_response_null_result(void **state)
{
  (void)state;
  char *resp = dt_server_make_response("1", NULL);
  assert_non_null(resp);

  JsonParser *p = json_parser_new();
  assert_true(json_parser_load_from_data(p, resp, -1, NULL));
  JsonObject *obj = json_node_get_object(json_parser_get_root(p));

  assert_true(json_object_get_null_member(obj, "result"));

  g_object_unref(p);
  g_free(resp);
}

static void test_make_error_method_not_found(void **state)
{
  (void)state;
  char *resp = dt_server_make_error("42", DT_SERVER_ERR_METHOD, "Method not found");
  assert_non_null(resp);

  JsonParser *p = json_parser_new();
  assert_true(json_parser_load_from_data(p, resp, -1, NULL));
  JsonObject *obj = json_node_get_object(json_parser_get_root(p));

  assert_string_equal(json_object_get_string_member(obj, "id"), "42");
  assert_true(json_object_get_null_member(obj, "result"));

  JsonObject *err = json_object_get_object_member(obj, "error");
  assert_non_null(err);
  assert_int_equal(json_object_get_int_member(err, "code"), -32601);
  assert_string_equal(json_object_get_string_member(err, "message"), "Method not found");

  g_object_unref(p);
  g_free(resp);
}

static void test_make_error_invalid_params(void **state)
{
  (void)state;
  char *resp = dt_server_make_error("5", DT_SERVER_ERR_PARAMS, "Invalid params");
  assert_non_null(resp);

  JsonParser *p = json_parser_new();
  assert_true(json_parser_load_from_data(p, resp, -1, NULL));
  JsonObject *err = json_object_get_object_member(
    json_node_get_object(json_parser_get_root(p)), "error");
  assert_int_equal(json_object_get_int_member(err, "code"), -32602);

  g_object_unref(p);
  g_free(resp);
}

static void test_make_error_parse_error(void **state)
{
  (void)state;
  char *resp = dt_server_make_error("", DT_SERVER_ERR_PARSE, "Parse error");
  assert_non_null(resp);

  JsonParser *p = json_parser_new();
  assert_true(json_parser_load_from_data(p, resp, -1, NULL));
  JsonObject *err = json_object_get_object_member(
    json_node_get_object(json_parser_get_root(p)), "error");
  assert_int_equal(json_object_get_int_member(err, "code"), -32700);

  g_object_unref(p);
  g_free(resp);
}

static void test_make_error_internal(void **state)
{
  (void)state;
  char *resp = dt_server_make_error("3", DT_SERVER_ERR_INTERNAL, "Internal error");
  assert_non_null(resp);

  JsonParser *p = json_parser_new();
  assert_true(json_parser_load_from_data(p, resp, -1, NULL));
  JsonObject *err = json_object_get_object_member(
    json_node_get_object(json_parser_get_root(p)), "error");
  assert_int_equal(json_object_get_int_member(err, "code"), -32603);

  g_object_unref(p);
  g_free(resp);
}

static void test_make_event(void **state)
{
  (void)state;
  JsonBuilder *b = json_builder_new();
  json_builder_begin_object(b);
  json_builder_set_member_name(b, "session_id");
  json_builder_add_string_value(b, "dev001");
  json_builder_end_object(b);
  JsonNode *data = json_builder_get_root(b);

  char *evt = dt_server_make_event("pipeline.finished", data);
  assert_non_null(evt);

  JsonParser *p = json_parser_new();
  assert_true(json_parser_load_from_data(p, evt, -1, NULL));
  JsonObject *obj = json_node_get_object(json_parser_get_root(p));

  assert_true(json_object_get_null_member(obj, "id"));
  assert_string_equal(json_object_get_string_member(obj, "event"), "pipeline.finished");

  JsonObject *d = json_object_get_object_member(obj, "data");
  assert_string_equal(json_object_get_string_member(d, "session_id"), "dev001");

  g_object_unref(p);
  g_free(evt);
  json_node_unref(data);
  g_object_unref(b);
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  const struct CMUnitTest tests[] = {
    /* Frame I/O */
    cmocka_unit_test(test_frame_roundtrip),
    cmocka_unit_test(test_frame_roundtrip_large),
    cmocka_unit_test(test_frame_multiple),
    cmocka_unit_test(test_frame_read_closed_fd),
    cmocka_unit_test(test_frame_zero_length_rejected),
    cmocka_unit_test(test_frame_oversized_rejected),
    /* Request parsing */
    cmocka_unit_test(test_parse_valid_request),
    cmocka_unit_test(test_parse_request_no_params),
    cmocka_unit_test(test_parse_request_missing_method),
    cmocka_unit_test(test_parse_request_invalid_json),
    cmocka_unit_test(test_parse_request_empty_string),
    cmocka_unit_test(test_parse_request_array_not_object),
    cmocka_unit_test(test_parse_request_no_id),
    /* Response / error / event building */
    cmocka_unit_test(test_make_response),
    cmocka_unit_test(test_make_response_null_result),
    cmocka_unit_test(test_make_error_method_not_found),
    cmocka_unit_test(test_make_error_invalid_params),
    cmocka_unit_test(test_make_error_parse_error),
    cmocka_unit_test(test_make_error_internal),
    cmocka_unit_test(test_make_event),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
