cmake_minimum_required(VERSION 3.18)

project(darktable_ai)

# Default to shared library to enforce ABI boundary
add_library(darktable_ai SHARED
  darktable_ai.c
  darktable_ai.h
)

# Define export macro when building the library
target_compile_definitions(darktable_ai PRIVATE DT_AI_EXPORTING)

if(APPLE)
  target_link_options(darktable_ai PRIVATE "-undefined" "dynamic_lookup")
endif()

# Find ONNX Runtime (auto-downloads if not present)
find_package(ONNXRuntime REQUIRED)

# Find Dependencies (inherited from main build if available, but explicit here for clarity)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED glib-2.0 gobject-2.0 gio-2.0 gmodule-2.0)
pkg_check_modules(JSON_GLIB REQUIRED json-glib-1.0)
# GTK3/RSVG needed for headers only (common/darktable.h -> utility.h -> gtk/gtk.h, librsvg/rsvg.h)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
pkg_check_modules(RSVG2 REQUIRED librsvg-2.0)

# Include current directory for header
target_include_directories(darktable_ai PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
  $<INSTALL_INTERFACE:include>
  ${GLIB_INCLUDE_DIRS}
  ${JSON_GLIB_INCLUDE_DIRS}
  ${GTK3_INCLUDE_DIRS}
  ${RSVG2_INCLUDE_DIRS}
)

target_link_directories(darktable_ai PRIVATE
  ${GLIB_LIBRARY_DIRS}
  ${JSON_GLIB_LIBRARY_DIRS}
)

target_link_libraries(darktable_ai PRIVATE
  onnxruntime::onnxruntime
  ${GLIB_LIBRARIES}
  ${JSON_GLIB_LIBRARIES}
)

# Set RPATH so darktable_ai can find libonnxruntime at runtime
if(APPLE)
  set_target_properties(darktable_ai PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_RPATH "${ONNXRuntime_LIB_DIR}"
  )
elseif(UNIX)
  set_target_properties(darktable_ai PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "${ONNXRuntime_LIB_DIR}"
  )
endif()

# Installation rules
install(TARGETS darktable_ai DESTINATION ${CMAKE_INSTALL_LIBDIR}/darktable)

# Install the bundled ONNX Runtime shared library alongside darktable_ai
if(APPLE)
  file(GLOB _ORT_DYLIB "${ONNXRuntime_LIB_DIR}/libonnxruntime.*.dylib")
  if(_ORT_DYLIB)
    install(FILES ${_ORT_DYLIB}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/darktable
      RENAME libonnxruntime.dylib
    )
  endif()
elseif(WIN32)
  file(GLOB _ORT_DLL "${ONNXRuntime_LIB_DIR}/onnxruntime.dll")
  if(_ORT_DLL)
    install(FILES ${_ORT_DLL}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/darktable
    )
  endif()
elseif(UNIX)
  file(GLOB _ORT_SO "${ONNXRuntime_LIB_DIR}/libonnxruntime.so*")
  if(_ORT_SO)
    install(FILES ${_ORT_SO}
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/darktable
    )
  endif()
endif()


