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
 * Transport latency benchmark.
 *
 * Measures the round-trip cost of the transport vtable's call() method
 * using two implementations:
 *
 *   1. Null transport — call() returns a static JSON string immediately.
 *      Measures pure vtable dispatch + timing overhead (the floor).
 *
 *   2. Loopback transport — socketpair with an echo responder thread.
 *      Uses the real frame I/O protocol (4-byte length prefix + JSON).
 *      Measures actual Unix socket round-trip latency.
 *
 * When real transport implementations (dt_transport_ipc_new,
 * dt_transport_direct_new) are available, add them as additional cases.
 *
 * Output: human-readable table + optional JSON (--json or --baseline).
 * Regression: compares against baseline.json if present (--baseline PATH).
 *
 * Usage:
 *   bench_transport [--iterations N] [--json] [--baseline PATH]
 */

#include "server/server_protocol.h"
#include "webview/transport.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

/* ── Timing helpers ───────────────────────────────────────────── */

static double _now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ── Benchmark result ─────────────────────────────────────────── */

typedef struct bench_result_t
{
  const char *name;
  int count;
  double min_us;
  double max_us;
  double p50_us;
  double p95_us;
  double p99_us;
  double mean_us;
  double throughput_rps; /* requests per second */
} bench_result_t;

static int _cmp_double(const void *a, const void *b)
{
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static bench_result_t _compute_stats(const char *name, double *samples, int n)
{
  bench_result_t r = { .name = name, .count = n };
  qsort(samples, n, sizeof(double), _cmp_double);

  /* samples are in nanoseconds; convert to microseconds for storage */
  r.min_us = samples[0] / 1e3;
  r.max_us = samples[n - 1] / 1e3;
  r.p50_us = samples[n / 2] / 1e3;
  r.p95_us = samples[(int)(n * 0.95)] / 1e3;
  r.p99_us = samples[(int)(n * 0.99)] / 1e3;

  double sum = 0;
  for(int i = 0; i < n; i++) sum += samples[i];
  r.mean_us = sum / n / 1e3;
  r.throughput_rps = 1e6 / r.mean_us;

  return r;
}

/* ── Null transport ───────────────────────────────────────────── */

static char *_null_call(dt_webview_transport_t *self,
                        const char *method,
                        const char *params_json,
                        char **error)
{
  (void)self; (void)method; (void)params_json; (void)error;
  return g_strdup("{\"status\":\"ok\"}");
}

static void _null_destroy(dt_webview_transport_t *self)
{
  g_free(self);
}

static dt_webview_transport_t *_null_transport_new(void)
{
  dt_webview_transport_t *t = g_new0(dt_webview_transport_t, 1);
  t->call = _null_call;
  t->destroy = _null_destroy;
  return t;
}

/* ── Loopback transport ───────────────────────────────────────── */

/* Echo responder thread: reads JSON-RPC requests from one end of
 * a socketpair, extracts the "id" field, and writes back a
 * well-formed response using the real frame I/O protocol. */

typedef struct loopback_data_t
{
  int client_fd;  /* benchmark thread uses this end */
  int server_fd;  /* echo thread uses this end */
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

    /* Parse just the "id" field */
    dt_server_request_t *req = dt_server_parse_request(buf, len);
    g_free(buf);
    buf = NULL;

    if(!req) continue;

    /* Build response with real protocol code */
    JsonBuilder *b = json_builder_new();
    json_builder_begin_object(b);
    json_builder_set_member_name(b, "status");
    json_builder_add_string_value(b, "ok");
    json_builder_end_object(b);
    JsonNode *result_node = json_builder_get_root(b);

    char *resp = dt_server_make_response(req->id, result_node);
    size_t resp_len = strlen(resp);

    dt_server_write_frame(lb->server_fd, resp, resp_len);

    json_node_unref(result_node);
    g_object_unref(b);
    g_free(resp);
    dt_server_free_request(req);
  }

  return NULL;
}

/* Loopback transport call(): send request via frame protocol,
 * wait for response. */
static char *_loopback_call(dt_webview_transport_t *self,
                            const char *method,
                            const char *params_json,
                            char **error)
{
  loopback_data_t *lb = self->data;

  /* Build JSON-RPC request */
  static int req_id = 0;
  char *request;
  if(params_json)
    request = g_strdup_printf("{\"id\":\"%d\",\"method\":\"%s\",\"params\":%s}",
                              ++req_id, method, params_json);
  else
    request = g_strdup_printf("{\"id\":\"%d\",\"method\":\"%s\"}", ++req_id, method);

  size_t req_len = strlen(request);

  if(!dt_server_write_frame(lb->client_fd, request, req_len))
  {
    g_free(request);
    if(error) *error = g_strdup("write failed");
    return NULL;
  }
  g_free(request);

  /* Read response */
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
  /* Close server fd to unblock the reader thread */
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
  {
    perror("socketpair");
    return NULL;
  }

  loopback_data_t *lb = g_new0(loopback_data_t, 1);
  lb->client_fd = fds[0];
  lb->server_fd = fds[1];
  lb->running = TRUE;

  if(pthread_create(&lb->thread, NULL, _echo_responder, lb) != 0)
  {
    perror("pthread_create");
    close(fds[0]);
    close(fds[1]);
    g_free(lb);
    return NULL;
  }

  dt_webview_transport_t *t = g_new0(dt_webview_transport_t, 1);
  t->data = lb;
  t->call = _loopback_call;
  t->destroy = _loopback_destroy;
  return t;
}

/* ── Frame delivery simulation ─────────────────────────────────── */

/* Simulates frame delivery: allocate + memset a frame buffer (like
 * the pipeline would produce) and memcpy it (like the transport would
 * deliver). Measures the copy cost without actual pipeline processing. */

typedef struct frame_bench_result_t
{
  int width;
  int height;
  double alloc_copy_us;    /* mean time: alloc + memcpy + free */
  double copy_only_us;     /* mean time: memcpy only */
  double fps;              /* estimated frames/sec from copy-only path */
} frame_bench_result_t;

static frame_bench_result_t _run_frame_benchmark(int width, int height, int iterations)
{
  const size_t frame_size = (size_t)width * height * 4; /* BGRA */
  frame_bench_result_t r = { .width = width, .height = height };

  /* Source buffer simulating pipeline backbuf */
  uint8_t *src = g_malloc(frame_size);
  memset(src, 0x42, frame_size);

  /* Pre-allocated destination for copy-only test */
  uint8_t *dst_prealloc = g_malloc(frame_size);

  double sum_full = 0, sum_copy = 0;

  for(int i = 0; i < iterations; i++)
  {
    /* Full path: alloc + copy + free (like direct transport get_preview_frame) */
    double t0 = _now_ns();
    uint8_t *dst = g_malloc(frame_size);
    memcpy(dst, src, frame_size);
    g_free(dst);
    double t1 = _now_ns();
    sum_full += (t1 - t0);

    /* Copy-only path (like in-place update) */
    t0 = _now_ns();
    memcpy(dst_prealloc, src, frame_size);
    t1 = _now_ns();
    sum_copy += (t1 - t0);
  }

  r.alloc_copy_us = sum_full / iterations / 1e3;
  r.copy_only_us = sum_copy / iterations / 1e3;
  r.fps = 1e6 / r.alloc_copy_us; /* based on full path */

  g_free(src);
  g_free(dst_prealloc);
  return r;
}


/* ── Run benchmark ────────────────────────────────────────────── */

static bench_result_t _run_benchmark(const char *name,
                                     dt_webview_transport_t *t,
                                     int iterations)
{
  double *samples = g_new(double, iterations);

  /* Warm up: 100 iterations (or fewer if total iterations is small) */
  int warmup = iterations < 200 ? iterations / 10 : 100;
  for(int i = 0; i < warmup; i++)
  {
    char *r = dt_transport_call(t, "system.ping", NULL, NULL);
    g_free(r);
  }

  /* Timed run */
  for(int i = 0; i < iterations; i++)
  {
    double t0 = _now_ns();
    char *r = dt_transport_call(t, "develop.set_params",
               "{\"session_id\":\"bench\",\"op\":\"exposure\","
               "\"multi_instance\":0,\"params\":{\"exposure\":1.5},"
               "\"preview_only\":true}", NULL);
    double t1 = _now_ns();
    g_free(r);
    samples[i] = t1 - t0;
  }

  bench_result_t result = _compute_stats(name, samples, iterations);
  g_free(samples);
  return result;
}

/* ── Output ───────────────────────────────────────────────────── */

static void _print_table(bench_result_t *results, int n)
{
  printf("\n%-24s %10s %10s %10s %10s %10s %12s\n",
         "Transport", "min(us)", "p50(us)", "p95(us)", "p99(us)", "max(us)", "throughput");
  printf("%-24s %10s %10s %10s %10s %10s %12s\n",
         "------------------------", "----------", "----------",
         "----------", "----------", "----------", "------------");

  for(int i = 0; i < n; i++)
  {
    bench_result_t *r = &results[i];
    printf("%-24s %10.3f %10.3f %10.3f %10.3f %10.3f %10.0f rps\n",
           r->name, r->min_us, r->p50_us, r->p95_us, r->p99_us, r->max_us,
           r->throughput_rps);
  }
  printf("\n");
}

static void _print_json(bench_result_t *results, int n, FILE *out)
{
  fprintf(out, "{\n");
  for(int i = 0; i < n; i++)
  {
    bench_result_t *r = &results[i];
    fprintf(out, "  \"%s\": {\n", r->name);
    fprintf(out, "    \"count\": %d,\n", r->count);
    fprintf(out, "    \"min_us\": %.2f,\n", r->min_us);
    fprintf(out, "    \"p50_us\": %.2f,\n", r->p50_us);
    fprintf(out, "    \"p95_us\": %.2f,\n", r->p95_us);
    fprintf(out, "    \"p99_us\": %.2f,\n", r->p99_us);
    fprintf(out, "    \"max_us\": %.2f,\n", r->max_us);
    fprintf(out, "    \"mean_us\": %.2f,\n", r->mean_us);
    fprintf(out, "    \"throughput_rps\": %.0f\n", r->throughput_rps);
    fprintf(out, "  }%s\n", i < n - 1 ? "," : "");
  }
  fprintf(out, "}\n");
}

/* ── Baseline comparison ──────────────────────────────────────── */

static int _check_baseline(const char *path, bench_result_t *results, int n,
                            double threshold_pct)
{
  /* Read baseline JSON */
  char *contents = NULL;
  size_t len = 0;
  if(!g_file_get_contents(path, &contents, &len, NULL))
  {
    fprintf(stderr, "[bench] baseline file not found: %s (skipping regression check)\n", path);
    return 0;
  }

  JsonParser *parser = json_parser_new();
  if(!json_parser_load_from_data(parser, contents, len, NULL))
  {
    fprintf(stderr, "[bench] baseline JSON parse error\n");
    g_free(contents);
    g_object_unref(parser);
    return 1;
  }

  JsonNode *root = json_parser_get_root(parser);
  if(!root || !JSON_NODE_HOLDS_OBJECT(root))
  {
    g_free(contents);
    g_object_unref(parser);
    return 1;
  }

  JsonObject *obj = json_node_get_object(root);
  int failures = 0;

  for(int i = 0; i < n; i++)
  {
    if(!json_object_has_member(obj, results[i].name)) continue;

    JsonObject *baseline = json_object_get_object_member(obj, results[i].name);
    if(!baseline) continue;

    /* Compare p99 latency */
    if(json_object_has_member(baseline, "p99_us"))
    {
      double base_p99 = json_object_get_double_member(baseline, "p99_us");
      double pct = (results[i].p99_us - base_p99) / base_p99 * 100.0;
      if(pct > threshold_pct)
      {
        fprintf(stderr, "[REGRESSION] %s p99: %.1f us → %.1f us (+%.0f%%, threshold %.0f%%)\n",
                results[i].name, base_p99, results[i].p99_us, pct, threshold_pct);
        failures++;
      }
      else
      {
        printf("[OK] %s p99: %.1f us (baseline %.1f us, %+.0f%%)\n",
               results[i].name, results[i].p99_us, base_p99, pct);
      }
    }

    /* Compare throughput (regression = lower throughput) */
    if(json_object_has_member(baseline, "throughput_rps"))
    {
      double base_thr = json_object_get_double_member(baseline, "throughput_rps");
      double pct = (base_thr - results[i].throughput_rps) / base_thr * 100.0;
      if(pct > threshold_pct)
      {
        fprintf(stderr, "[REGRESSION] %s throughput: %.0f → %.0f rps (-%.0f%%, threshold %.0f%%)\n",
                results[i].name, base_thr, results[i].throughput_rps, pct, threshold_pct);
        failures++;
      }
    }
  }

  g_free(contents);
  g_object_unref(parser);
  return failures;
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
  int iterations = 10000;
  gboolean json_output = FALSE;
  const char *baseline_path = NULL;
  double threshold_pct = 20.0;

  for(int i = 1; i < argc; i++)
  {
    if(!strcmp(argv[i], "--iterations") && i + 1 < argc)
      iterations = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--json"))
      json_output = TRUE;
    else if(!strcmp(argv[i], "--baseline") && i + 1 < argc)
      baseline_path = argv[++i];
    else if(!strcmp(argv[i], "--threshold") && i + 1 < argc)
      threshold_pct = atof(argv[++i]);
    else if(!strcmp(argv[i], "--help"))
    {
      printf("Usage: bench_transport [OPTIONS]\n"
             "  --iterations N   Number of calls per benchmark (default: 10000)\n"
             "  --json           Output results as JSON to stdout\n"
             "  --baseline PATH  Compare against baseline JSON file\n"
             "  --threshold PCT  Regression threshold percentage (default: 20)\n");
      return 0;
    }
  }

  printf("NOVA Transport Benchmark\n");
  printf("Iterations: %d\n", iterations);

  /* --- Null transport --- */
  dt_webview_transport_t *null_t = _null_transport_new();
  bench_result_t null_result = _run_benchmark("null", null_t, iterations);
  dt_transport_destroy(null_t);

  /* --- Loopback transport --- */
  dt_webview_transport_t *loopback_t = _loopback_transport_new();
  if(!loopback_t)
  {
    fprintf(stderr, "[bench] failed to create loopback transport\n");
    return 1;
  }
  bench_result_t loopback_result = _run_benchmark("loopback", loopback_t, iterations);
  dt_transport_destroy(loopback_t);

  /* --- Frame delivery benchmark --- */
  int frame_iters = iterations < 1000 ? iterations : 1000;
  frame_bench_result_t frame_720  = _run_frame_benchmark(1280, 720,  frame_iters);
  frame_bench_result_t frame_1080 = _run_frame_benchmark(1920, 1080, frame_iters);
  frame_bench_result_t frame_4k   = _run_frame_benchmark(3840, 2160, frame_iters);

  /* --- Output --- */
  bench_result_t results[] = { null_result, loopback_result };
  int n_results = 2;

  if(json_output)
  {
    _print_json(results, n_results, stdout);
  }
  else
  {
    _print_table(results, n_results);

    /* Overhead ratio */
    if(null_result.mean_us > 0.001)
      printf("Socket overhead: %.0fx (loopback mean / null mean)\n\n",
             loopback_result.mean_us / null_result.mean_us);
    else
      printf("Null transport: < 1 ns/call (below timer resolution)\n\n");

    /* Frame delivery table */
    printf("Frame Delivery (alloc + memcpy + free)\n");
    printf("%-20s %10s %10s %10s %10s\n",
           "Resolution", "Size(MB)", "Copy(us)", "Full(us)", "Max FPS");
    printf("%-20s %10s %10s %10s %10s\n",
           "--------------------", "----------", "----------",
           "----------", "----------");
    frame_bench_result_t frames[] = { frame_720, frame_1080, frame_4k };
    const char *labels[] = { "1280x720", "1920x1080", "3840x2160" };
    for(int i = 0; i < 3; i++)
    {
      double size_mb = (double)frames[i].width * frames[i].height * 4 / (1024.0 * 1024.0);
      printf("%-20s %10.1f %10.1f %10.1f %10.0f\n",
             labels[i], size_mb, frames[i].copy_only_us,
             frames[i].alloc_copy_us, frames[i].fps);
    }
    printf("\n");
  }

  /* --- Baseline regression check --- */
  int failures = 0;
  if(baseline_path)
    failures = _check_baseline(baseline_path, results, n_results, threshold_pct);

  return failures > 0 ? 1 : 0;
}
