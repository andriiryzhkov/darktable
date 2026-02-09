# FindONNXRuntime.cmake
#
# Find ONNX Runtime pre-built binaries, downloading them automatically if not
# found. Searches two locations:
#
#   1. Source tree:  ${CMAKE_SOURCE_DIR}/src/external/onnxruntime/
#      (manually pre-installed)
#   2. Build tree:   ${CMAKE_BINARY_DIR}/_deps/onnxruntime/
#      (auto-download destination)
#
# After this module completes the imported target onnxruntime::onnxruntime is
# available for linking, along with the standard variables:
#
#   ONNXRuntime_FOUND          - TRUE if found
#   ONNXRuntime_INCLUDE_DIRS   - include directories
#   ONNXRuntime_LIBRARIES      - libraries to link
#   ONNXRuntime_LIB_DIR        - directory containing the shared library
#
# Cache variables that influence behaviour:
#
#   ONNXRUNTIME_VERSION   - version to download (default 1.23.2)
#   ONNXRUNTIME_OFFLINE   - if TRUE, never attempt a download (default OFF)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_VERSION "1.23.2" CACHE STRING "ONNX Runtime version to download")
option(ONNXRUNTIME_OFFLINE "Disable automatic download of ONNX Runtime" OFF)

set(_ORT_SRC_ROOT "${CMAKE_SOURCE_DIR}/src/external/onnxruntime")
set(_ORT_BUILD_ROOT "${CMAKE_BINARY_DIR}/_deps/onnxruntime")

# ---------------------------------------------------------------------------
# Search for existing installation (source tree first, then build tree)
# ---------------------------------------------------------------------------
macro(_ort_find_at ROOT)
  find_path(_ORT_HEADER
    NAMES onnxruntime_c_api.h
    PATHS "${ROOT}/include"
    NO_DEFAULT_PATH
  )
  find_library(_ORT_LIBRARY
    NAMES onnxruntime
    PATHS "${ROOT}/lib"
    NO_DEFAULT_PATH
  )
  if(_ORT_HEADER AND _ORT_LIBRARY)
    set(_ORT_ROOT "${ROOT}")
  endif()
endmacro()

# Try source tree (manually pre-installed)
_ort_find_at("${_ORT_SRC_ROOT}")

# Try build tree (populated by prior auto-download)
if(NOT _ORT_HEADER OR NOT _ORT_LIBRARY)
  unset(_ORT_HEADER CACHE)
  unset(_ORT_LIBRARY CACHE)
  _ort_find_at("${_ORT_BUILD_ROOT}")
endif()

# ---------------------------------------------------------------------------
# Auto-download if not found
# ---------------------------------------------------------------------------
if(NOT _ORT_HEADER OR NOT _ORT_LIBRARY)
  if(ONNXRUNTIME_OFFLINE)
    message(FATAL_ERROR
      "ONNX Runtime not found and ONNXRUNTIME_OFFLINE is ON.\n"
      "Please install ONNX Runtime manually to src/external/onnxruntime/ or set ONNXRUNTIME_OFFLINE=OFF.")
  endif()

  # -- Determine package name for current platform --
  set(_ORT_VER "${ONNXRUNTIME_VERSION}")

  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
      set(_ORT_PACKAGE "onnxruntime-osx-arm64-${_ORT_VER}.tgz")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
      set(_ORT_PACKAGE "onnxruntime-osx-x86_64-${_ORT_VER}.tgz")
    else()
      message(FATAL_ERROR "Unsupported macOS architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
      set(_ORT_PACKAGE "onnxruntime-linux-x64-${_ORT_VER}.tgz")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
      set(_ORT_PACKAGE "onnxruntime-linux-aarch64-${_ORT_VER}.tgz")
    else()
      message(FATAL_ERROR "Unsupported Linux architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
      set(_ORT_PACKAGE "onnxruntime-win-x64-${_ORT_VER}.zip")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
      set(_ORT_PACKAGE "onnxruntime-win-arm64-${_ORT_VER}.zip")
    else()
      message(FATAL_ERROR "Unsupported Windows architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
  else()
    message(FATAL_ERROR "Unsupported OS for ONNX Runtime auto-download: ${CMAKE_SYSTEM_NAME}")
  endif()

  set(_ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${_ORT_VER}/${_ORT_PACKAGE}")
  set(_ORT_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_deps")
  set(_ORT_ARCHIVE "${_ORT_DOWNLOAD_DIR}/${_ORT_PACKAGE}")

  # -- Fetch SHA256 digest from GitHub Releases API --
  # GitHub provides a "digest" field (sha256:...) for every release asset.
  set(_ORT_HASH "")
  set(_ORT_API_JSON "${_ORT_DOWNLOAD_DIR}/ort-release-${_ORT_VER}.json")
  set(_ORT_API_URL "https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v${_ORT_VER}")

  if(NOT EXISTS "${_ORT_API_JSON}")
    message(STATUS "Fetching ONNX Runtime v${_ORT_VER} release metadata from GitHub API...")
    file(MAKE_DIRECTORY "${_ORT_DOWNLOAD_DIR}")
    file(DOWNLOAD
      "${_ORT_API_URL}"
      "${_ORT_API_JSON}"
      STATUS _ORT_API_STATUS
      HTTPHEADER "Accept: application/vnd.github+json"
    )
    list(GET _ORT_API_STATUS 0 _ORT_API_CODE)
    if(NOT _ORT_API_CODE EQUAL 0)
      file(REMOVE "${_ORT_API_JSON}")
    endif()
  endif()

  if(EXISTS "${_ORT_API_JSON}")
    file(READ "${_ORT_API_JSON}" _ORT_API_CONTENT)
    # Two-step extraction: locate the package name, then find the first
    # "digest" field that follows it within the same asset JSON object.
    string(FIND "${_ORT_API_CONTENT}" "\"name\":\"${_ORT_PACKAGE}\"" _ORT_NAME_POS)
    if(_ORT_NAME_POS EQUAL -1)
      # Try with spaces around colon (GitHub API may format either way)
      string(FIND "${_ORT_API_CONTENT}" "\"name\": \"${_ORT_PACKAGE}\"" _ORT_NAME_POS)
    endif()
    if(NOT _ORT_NAME_POS EQUAL -1)
      string(SUBSTRING "${_ORT_API_CONTENT}" ${_ORT_NAME_POS} 2000 _ORT_ASSET_TAIL)
      string(REGEX MATCH "\"digest\" *: *\"sha256:([a-f0-9]+)\"" _ORT_DIGEST_MATCH "${_ORT_ASSET_TAIL}")
      if(_ORT_DIGEST_MATCH)
        set(_ORT_HASH "${CMAKE_MATCH_1}")
        message(STATUS "ONNX Runtime ${_ORT_PACKAGE} SHA256: ${_ORT_HASH}")
      endif()
    endif()
    if(NOT _ORT_HASH)
      message(WARNING
        "Could not find SHA256 digest for ${_ORT_PACKAGE} in GitHub API response. "
        "Download will proceed without integrity verification.")
    endif()
  else()
    message(WARNING
      "Could not fetch release metadata from GitHub API. "
      "Download will proceed without integrity verification.")
  endif()

  # -- Verify cached archive if it exists --
  if(EXISTS "${_ORT_ARCHIVE}" AND _ORT_HASH)
    file(SHA256 "${_ORT_ARCHIVE}" _ORT_CACHED_HASH)
    if(NOT _ORT_CACHED_HASH STREQUAL "${_ORT_HASH}")
      message(STATUS "Cached ONNX Runtime archive has wrong checksum, re-downloading...")
      file(REMOVE "${_ORT_ARCHIVE}")
    endif()
  endif()

  # -- Download --
  if(NOT EXISTS "${_ORT_ARCHIVE}")
    message(STATUS "Downloading ONNX Runtime ${_ORT_VER} (${_ORT_PACKAGE})...")
    file(MAKE_DIRECTORY "${_ORT_DOWNLOAD_DIR}")
    if(_ORT_HASH)
      file(DOWNLOAD
        "${_ORT_URL}"
        "${_ORT_ARCHIVE}"
        STATUS _ORT_DL_STATUS
        SHOW_PROGRESS
        EXPECTED_HASH "SHA256=${_ORT_HASH}"
      )
    else()
      file(DOWNLOAD
        "${_ORT_URL}"
        "${_ORT_ARCHIVE}"
        STATUS _ORT_DL_STATUS
        SHOW_PROGRESS
      )
    endif()
    list(GET _ORT_DL_STATUS 0 _ORT_DL_CODE)
    list(GET _ORT_DL_STATUS 1 _ORT_DL_MSG)
    if(NOT _ORT_DL_CODE EQUAL 0)
      file(REMOVE "${_ORT_ARCHIVE}")
      message(FATAL_ERROR
        "Failed to download ONNX Runtime from ${_ORT_URL}\n"
        "Error: ${_ORT_DL_MSG}\n"
        "You can download manually from: https://github.com/microsoft/onnxruntime/releases")
    endif()
  endif()

  # -- Extract to a temporary directory --
  set(_ORT_EXTRACT_DIR "${_ORT_DOWNLOAD_DIR}/onnxruntime-extract")
  if(EXISTS "${_ORT_EXTRACT_DIR}")
    file(REMOVE_RECURSE "${_ORT_EXTRACT_DIR}")
  endif()
  message(STATUS "Extracting ${_ORT_PACKAGE}...")
  file(ARCHIVE_EXTRACT
    INPUT "${_ORT_ARCHIVE}"
    DESTINATION "${_ORT_EXTRACT_DIR}"
  )

  # Find the single top-level directory inside the extracted archive
  file(GLOB _ORT_INNER_DIRS "${_ORT_EXTRACT_DIR}/*")
  list(LENGTH _ORT_INNER_DIRS _ORT_INNER_COUNT)
  if(_ORT_INNER_COUNT EQUAL 1)
    list(GET _ORT_INNER_DIRS 0 _ORT_INNER)
  else()
    set(_ORT_INNER "${_ORT_EXTRACT_DIR}")
  endif()

  # -- Install into build tree --
  if(EXISTS "${_ORT_BUILD_ROOT}")
    file(REMOVE_RECURSE "${_ORT_BUILD_ROOT}")
  endif()
  file(MAKE_DIRECTORY "${_ORT_BUILD_ROOT}")
  file(COPY "${_ORT_INNER}/lib" DESTINATION "${_ORT_BUILD_ROOT}")
  file(COPY "${_ORT_INNER}/include" DESTINATION "${_ORT_BUILD_ROOT}")

  # -- Cleanup extraction directory --
  file(REMOVE_RECURSE "${_ORT_EXTRACT_DIR}")

  # -- Set root and re-search --
  set(_ORT_ROOT "${_ORT_BUILD_ROOT}")

  unset(_ORT_HEADER CACHE)
  unset(_ORT_LIBRARY CACHE)

  find_path(_ORT_HEADER
    NAMES onnxruntime_c_api.h
    PATHS "${_ORT_ROOT}/include"
    NO_DEFAULT_PATH
  )
  find_library(_ORT_LIBRARY
    NAMES onnxruntime
    PATHS "${_ORT_ROOT}/lib"
    NO_DEFAULT_PATH
  )
endif()

# ---------------------------------------------------------------------------
# Create onnxruntime::onnxruntime imported target
# ---------------------------------------------------------------------------
if(_ORT_HEADER AND _ORT_LIBRARY AND NOT TARGET onnxruntime::onnxruntime)
  # Try the shipped CMake config files first (they exist under lib/cmake/)
  set(_ORT_CMAKE_DIR "${_ORT_ROOT}/lib/cmake/onnxruntime")
  set(_ORT_CMAKE_TARGETS "${_ORT_CMAKE_DIR}/onnxruntimeTargets.cmake")

  # Patch the known include-path bug before loading the config
  if(EXISTS "${_ORT_CMAKE_TARGETS}")
    file(READ "${_ORT_CMAKE_TARGETS}" _ORT_TARGETS_CONTENT)
    if(_ORT_TARGETS_CONTENT MATCHES "include/onnxruntime"
       AND NOT EXISTS "${_ORT_ROOT}/include/onnxruntime")
      string(REPLACE "/include/onnxruntime" "/include"
        _ORT_TARGETS_CONTENT "${_ORT_TARGETS_CONTENT}")
      file(WRITE "${_ORT_CMAKE_TARGETS}" "${_ORT_TARGETS_CONTENT}")
      message(STATUS "Patched ONNX Runtime CMake config: include/onnxruntime -> include")
    endif()
  endif()

  # Patch the lib64 vs lib path issue in CMake targets
  file(GLOB _ORT_CMAKE_TARGET_FILES "${_ORT_CMAKE_DIR}/onnxruntimeTargets*.cmake")
  foreach(_ORT_TARGET_FILE ${_ORT_CMAKE_TARGET_FILES})
    if(EXISTS "${_ORT_TARGET_FILE}")
      file(READ "${_ORT_TARGET_FILE}" _ORT_TARGET_CONTENT)
      if(_ORT_TARGET_CONTENT MATCHES "/lib64/"
         AND NOT EXISTS "${_ORT_ROOT}/lib64")
        string(REPLACE "/lib64/" "/lib/" _ORT_TARGET_CONTENT "${_ORT_TARGET_CONTENT}")
        file(WRITE "${_ORT_TARGET_FILE}" "${_ORT_TARGET_CONTENT}")
        message(STATUS "Patched ONNX Runtime CMake config: lib64 -> lib")
      endif()
    endif()
  endforeach()

  find_package(onnxruntime QUIET
    PATHS "${_ORT_ROOT}"
    NO_DEFAULT_PATH
  )

  # Fallback: create the imported target manually
  if(NOT TARGET onnxruntime::onnxruntime)
    message(STATUS "Creating onnxruntime::onnxruntime imported target manually")
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_ORT_ROOT}/include"
    )
    if(WIN32)
      # On Windows find_library finds the .lib import library;
      # IMPORTED_IMPLIB = .lib, IMPORTED_LOCATION = .dll
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${_ORT_LIBRARY}"
      )
      find_file(_ORT_DLL NAMES onnxruntime.dll
        PATHS "${_ORT_ROOT}/lib"
        NO_DEFAULT_PATH
      )
      if(_ORT_DLL)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
          IMPORTED_LOCATION "${_ORT_DLL}"
        )
      endif()
    else()
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${_ORT_LIBRARY}"
      )
      if(APPLE)
        get_filename_component(_ORT_LIB_NAME "${_ORT_LIBRARY}" NAME)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
          IMPORTED_SONAME "@rpath/${_ORT_LIB_NAME}"
        )
      endif()
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Standard find_package result handling
# ---------------------------------------------------------------------------
mark_as_advanced(_ORT_HEADER _ORT_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
  REQUIRED_VARS _ORT_LIBRARY _ORT_HEADER
)

if(ONNXRuntime_FOUND)
  set(ONNXRuntime_INCLUDE_DIRS "${_ORT_ROOT}/include")
  set(ONNXRuntime_LIBRARIES "${_ORT_LIBRARY}")
  set(ONNXRuntime_LIB_DIR "${_ORT_ROOT}/lib" CACHE PATH "ONNX Runtime library directory")
  mark_as_advanced(ONNXRuntime_LIB_DIR)
endif()
