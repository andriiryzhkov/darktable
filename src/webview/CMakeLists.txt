# Build webview static library from submodule (disable everything we don't need)
set(WEBVIEW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WEBVIEW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WEBVIEW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(WEBVIEW_BUILD_AMALGAMATION OFF CACHE BOOL "" FORCE)
set(WEBVIEW_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
set(WEBVIEW_BUILD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(WEBVIEW_INSTALL_TARGETS OFF CACHE BOOL "" FORCE)
set(WEBVIEW_ENABLE_CHECKS OFF CACHE BOOL "" FORCE)
set(WEBVIEW_ENABLE_PACKAGING OFF CACHE BOOL "" FORCE)

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../external/webview
  ${CMAKE_CURRENT_BINARY_DIR}/webview_build
  EXCLUDE_FROM_ALL
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0)

set(WEBVIEW_HOST_SOURCES
  main.c
  ipc_client.c
  webview_bindings.c
  ../server/server_protocol.c
)

add_executable(darktable-webview ${WEBVIEW_HOST_SOURCES})

# We include C API headers only (api.h, types.h, errors.h) — no C++ in our code.
# Linking to webview_core_static provides the C++ implementation.
target_include_directories(darktable-webview PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..        # for "server/server_protocol.h"
  ${GLIB_INCLUDE_DIRS}
  ${JSON_GLIB_INCLUDE_DIRS}
)

# webview is C++ internally, so we need C++ linker
set_target_properties(darktable-webview PROPERTIES LINKER_LANGUAGE CXX)

target_link_directories(darktable-webview PRIVATE
  ${GLIB_LIBRARY_DIRS}
  ${JSON_GLIB_LIBRARY_DIRS}
)

target_link_libraries(darktable-webview
  webview_core_static
  ${GLIB_LIBRARIES}
  ${JSON_GLIB_LIBRARIES}
  pthread
)

if(APPLE)
  target_link_libraries(darktable-webview "-framework WebKit" "-framework Cocoa")
endif()

set_target_properties(darktable-webview
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${DARKTABLE_BINDIR}
)

install(TARGETS darktable-webview DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT DTApplication)
