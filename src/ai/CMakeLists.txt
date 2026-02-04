cmake_minimum_required(VERSION 3.18)

project(darktable_ai)

# Default to shared library to enforce ABI boundary
add_library(darktable_ai SHARED
  darktable_ai.c
  darktable_ai.h
)

# Define export macro when building the library
# Define export macro when building the library
target_compile_definitions(darktable_ai PRIVATE DT_AI_EXPORTING)

if(APPLE)
  target_link_options(darktable_ai PRIVATE "-undefined" "dynamic_lookup")
endif()

# Find ONNX Runtime
# Use the bundled GitHub release (with CoreML support) instead of Homebrew
find_package(onnxruntime
  PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../external/onnxruntime"
  NO_DEFAULT_PATH
  REQUIRED
)

# Find Dependencies (inherited from main build if available, but explicit here for clarity)
# usage of PkgConfig is common in Darktable for these
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED glib-2.0 gobject-2.0 gio-2.0 gmodule-2.0)
pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
pkg_check_modules(RSVG2 REQUIRED librsvg-2.0)
pkg_check_modules(PANGO REQUIRED pango)
pkg_check_modules(CAIRO REQUIRED cairo)
pkg_check_modules(GDK_PIXBUF REQUIRED gdk-pixbuf-2.0)

# Include current directory for header
target_include_directories(darktable_ai PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
  $<INSTALL_INTERFACE:include>
  ${GLIB_INCLUDE_DIRS}
  ${JSON_GLIB_INCLUDE_DIRS}
  ${JSON_GLIB_INCLUDE_DIRS}
)

target_link_directories(darktable_ai PRIVATE
  ${GLIB_LIBRARY_DIRS}
  ${JSON_GLIB_LIBRARY_DIRS}
  ${GTK3_LIBRARY_DIRS}
  ${RSVG2_LIBRARY_DIRS}
  ${PANGO_LIBRARY_DIRS}
  ${CAIRO_LIBRARY_DIRS}
  ${GDK_PIXBUF_LIBRARY_DIRS}
)

target_link_libraries(darktable_ai PRIVATE 
  onnxruntime::onnxruntime
  ${GLIB_LIBRARIES}
  ${JSON_GLIB_LIBRARIES}
  ${GTK3_LIBRARIES}
  ${RSVG2_LIBRARIES}
  ${PANGO_LIBRARIES}
  ${CAIRO_LIBRARIES}
  ${GDK_PIXBUF_LIBRARIES}
)

target_include_directories(darktable_ai PUBLIC
  ${GTK3_INCLUDE_DIRS}
  ${RSVG2_INCLUDE_DIRS}
  ${PANGO_INCLUDE_DIRS}
  ${CAIRO_INCLUDE_DIRS}
  ${GDK_PIXBUF_INCLUDE_DIRS}
)

# Set RPATH so darktable_ai can find libonnxruntime at runtime
set_target_properties(darktable_ai PROPERTIES
  INSTALL_RPATH "@loader_path"
  BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../external/onnxruntime/lib"
)

# Installation rules
install(TARGETS darktable_ai DESTINATION ${CMAKE_INSTALL_LIBDIR}/darktable)

# Install the bundled ONNX Runtime dylib alongside darktable_ai
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/onnxruntime/lib/libonnxruntime.1.23.2.dylib"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/darktable
  RENAME libonnxruntime.dylib
)


