find_package(PkgConfig REQUIRED)
pkg_check_modules(GOBJECT REQUIRED gobject-2.0)

add_executable(bench_transport
  bench_transport.c
  ${CMAKE_SOURCE_DIR}/src/server/server_protocol.c)

target_include_directories(bench_transport PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${Glib_INCLUDE_DIR}
  ${GlibConfig_INCLUDE_DIR}
  ${JsonGlib_INCLUDE_DIR})

target_link_libraries(bench_transport PRIVATE
  ${Glib_LIBRARY}
  ${JsonGlib_LIBRARY}
  ${GOBJECT_LIBRARIES})

target_link_directories(bench_transport PRIVATE
  ${GOBJECT_LIBRARY_DIRS})

# Register as a labeled test so it can be run separately:
#   ctest -L perf
add_test(NAME bench_transport
         COMMAND bench_transport --iterations 10000)
set_tests_properties(bench_transport PROPERTIES
  LABELS "perf"
  TIMEOUT 120)
