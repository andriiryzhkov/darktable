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

#include <limits.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include <cmocka.h>

/* Include the shared path validation header directly */
#include "webview/path_validation.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

/* ── dt_path_is_allowed tests ──────────────────────────────── */

static void test_null_path_rejected(void **state)
{
  (void)state;
  assert_false(dt_path_is_allowed(NULL));
}

static void test_empty_path_rejected(void **state)
{
  (void)state;
  assert_false(dt_path_is_allowed(""));
}

static void test_home_dir_allowed(void **state)
{
  (void)state;
  const char *home = g_get_home_dir();
  assert_true(dt_path_is_allowed(home));

  char *subdir = g_build_filename(home, "Pictures", NULL);
  assert_true(dt_path_is_allowed(subdir));
  g_free(subdir);

  char *nested = g_build_filename(home, "Documents", "photos", NULL);
  assert_true(dt_path_is_allowed(nested));
  g_free(nested);
}

static void test_root_allowed(void **state)
{
  (void)state;
  assert_true(dt_path_is_allowed("/"));
}

static void test_etc_rejected(void **state)
{
  (void)state;
  assert_false(dt_path_is_allowed("/etc"));
  assert_false(dt_path_is_allowed("/etc/passwd"));
}

static void test_usr_rejected(void **state)
{
  (void)state;
  assert_false(dt_path_is_allowed("/usr"));
  assert_false(dt_path_is_allowed("/usr/bin"));
}

static void test_var_rejected(void **state)
{
  (void)state;
  assert_false(dt_path_is_allowed("/var"));
  assert_false(dt_path_is_allowed("/var/log"));
}

static void test_traversal_rejected(void **state)
{
  (void)state;
  const char *home = g_get_home_dir();

  /* Path that tries to escape home via /../ */
  char *traversal = g_strdup_printf("%s/../../../etc/passwd", home);
  assert_false(dt_path_is_allowed(traversal));
  g_free(traversal);
}

static void test_traversal_suffix_rejected(void **state)
{
  (void)state;
  const char *home = g_get_home_dir();

  /* Path ending with /.. */
  char *traversal = g_strdup_printf("%s/..", home);
  assert_false(dt_path_is_allowed(traversal));
  g_free(traversal);
}

#ifdef __APPLE__
static void test_volumes_allowed(void **state)
{
  (void)state;
  assert_true(dt_path_is_allowed("/Volumes/ExternalDrive"));
  assert_true(dt_path_is_allowed("/Volumes/USB/Photos"));
}
#else
static void test_media_allowed(void **state)
{
  (void)state;
  /* These paths don't exist but should pass prefix check */
  assert_true(dt_path_is_allowed("/media/user/drive"));
  assert_true(dt_path_is_allowed("/mnt/external"));
  assert_true(dt_path_is_allowed("/run/media/user/usb"));
}
#endif

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_null_path_rejected),
    cmocka_unit_test(test_empty_path_rejected),
    cmocka_unit_test(test_home_dir_allowed),
    cmocka_unit_test(test_root_allowed),
    cmocka_unit_test(test_etc_rejected),
    cmocka_unit_test(test_usr_rejected),
    cmocka_unit_test(test_var_rejected),
    cmocka_unit_test(test_traversal_rejected),
    cmocka_unit_test(test_traversal_suffix_rejected),
#ifdef __APPLE__
    cmocka_unit_test(test_volumes_allowed),
#else
    cmocka_unit_test(test_media_allowed),
#endif
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
