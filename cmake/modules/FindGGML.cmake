# - Find the GGML library and includes
#
# This module defines
#  GGML_INCLUDE_DIRS - where to find ggml.h and related headers
#  GGML_LIBRARIES    - libraries needed to use GGML
#  GGML_FOUND        - True if GGML found
#
# The module will first try to find the package using the CMake config files.
# If that fails, it will try to find the libraries and headers manually.

include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(GGML_PKGCONF ggml)

# Main include dir
find_path(GGML_INCLUDE_DIR
  NAMES ggml.h
  HINTS ${GGML_PKGCONF_INCLUDE_DIRS}
  PATH_SUFFIXES include llama.cpp/include llama_cpp/include
)
mark_as_advanced(GGML_INCLUDE_DIR)

find_path(GGML_INCLUDE_DIR
  NAMES ggml-alloc.h
  HINTS ${GGML_PKGCONF_INCLUDE_DIRS}
  PATH_SUFFIXES include llama.cpp/include llama_cpp/include
)

find_path(GGML_INCLUDE_DIR
  NAMES ggml-backend.h
  HINTS ${GGML_PKGCONF_INCLUDE_DIRS}
  PATH_SUFFIXES include llama.cpp/include llama_cpp/include
)

find_path(GGML_INCLUDE_DIR
  NAMES ggml-cpu.h
  HINTS ${GGML_PKGCONF_INCLUDE_DIRS}
  PATH_SUFFIXES include llama.cpp/include llama_cpp/include
)

# Find the core GGML library
find_library(GGML_LIBRARY
  NAMES ggml libggml
  HINTS ${GGML_PKGCONF_LIBRARY_DIRS}
  PATH_SUFFIXES lib llama.cpp/lib llama_cpp/lib build lib64
)
mark_as_advanced(GGML_LIBRARY)

find_library(GGML_BASE_LIBRARY
  NAMES ggml-base libggml-base
  HINTS ${GGML_PKGCONF_LIBRARY_DIRS}
  PATH_SUFFIXES lib llama.cpp/lib llama_cpp/lib build lib64
)
mark_as_advanced(GGML_BASE_LIBRARY)

find_library(GGML_BLAS_LIBRARY
  NAMES ggml-blas libggml-blas
  HINTS ${GGML_PKGCONF_LIBRARY_DIRS}
  PATH_SUFFIXES lib llama.cpp/lib llama_cpp/lib build lib64
)
mark_as_advanced(GGML_BLAS_LIBRARY)

find_library(GGML_CPU_LIBRARY
  NAMES ggml-cpu libggml-cpu
  HINTS ${GGML_PKGCONF_LIBRARY_DIRS}
  PATH_SUFFIXES lib llama.cpp/lib llama_cpp/lib build lib64
)
mark_as_advanced(GGML_CPU_LIBRARY)

# Find optional components
find_library(GGML_OPENCL_LIBRARY
  NAMES ggml-opencl libggml-opencl
  HINTS ${GGML_PKGCONF_LIBRARY_DIRS}
  PATH_SUFFIXES lib llama.cpp/lib llama_cpp/lib build lib64
)
mark_as_advanced(GGML_OPENCL_LIBRARY)

find_library(GGML_VULKAN_LIBRARY
  NAMES ggml-vulkan libggml-vulkan
  HINTS ${GGML_PKGCONF_LIBRARY_DIRS}
  PATH_SUFFIXES lib llama.cpp/lib llama_cpp/lib build lib64
)
mark_as_advanced(GGML_VULKAN_LIBRARY)

find_library(GGML_RPC_LIBRARY
  NAMES ggml-rpc libggml-rpc
  HINTS ${GGML_PKGCONF_LIBRARY_DIRS}
  PATH_SUFFIXES lib llama.cpp/lib llama_cpp/lib build lib64
)
mark_as_advanced(GGML_RPC_LIBRARY)

find_library(GGML_METAL_LIBRARY
  NAMES ggml-metal libggml-metal
  HINTS ${GGML_PKGCONF_LIBRARY_DIRS}
  PATH_SUFFIXES lib llama.cpp/lib llama_cpp/lib build lib64
)
mark_as_advanced(GGML_METAL_LIBRARY)

# Add all component libraries if found
set(GGML_COMPONENT_LIBRARIES "")
if(GGML_BASE_LIBRARY)
    list(APPEND GGML_COMPONENT_LIBRARIES ${GGML_BASE_LIBRARY})
endif()
if(GGML_BLAS_LIBRARY)
    list(APPEND GGML_COMPONENT_LIBRARIES ${GGML_BLAS_LIBRARY})
endif()
if(GGML_CPU_LIBRARY)
    list(APPEND GGML_COMPONENT_LIBRARIES ${GGML_CPU_LIBRARY})
endif()
if(GGML_OPENCL_LIBRARY)
    list(APPEND GGML_COMPONENT_LIBRARIES ${GGML_OPENCL_LIBRARY})
endif()
if(GGML_VULKAN_LIBRARY)
    list(APPEND GGML_COMPONENT_LIBRARIES ${GGML_VULKAN_LIBRARY})
endif()
if(GGML_RPC_LIBRARY)
    list(APPEND GGML_COMPONENT_LIBRARIES ${GGML_RPC_LIBRARY})
endif()
if(GGML_METAL_LIBRARY)
    list(APPEND GGML_COMPONENT_LIBRARIES ${GGML_METAL_LIBRARY})
endif()

# Apply standard rules to see if required components are found
include(FindPackageHandleStandardArgs)

# GGML requirements
find_package_handle_standard_args(GGML REQUIRED_VARS GGML_LIBRARY GGML_INCLUDE_DIR)

# Set libraries and include directories if GGML is found
if(GGML_FOUND)
    set(GGML_LIBRARIES ${GGML_LIBRARY} ${GGML_COMPONENT_LIBRARIES})
    set(GGML_INCLUDE_DIRS ${GGML_INCLUDE_DIR})
    
    # Feature detection - OpenCL support
    if(EXISTS "${GGML_HEADER_FILE}")
        file(STRINGS "${GGML_HEADER_FILE}" GGML_OPENCL_LINE REGEX "GGML_USE_OPENCL")
        if(GGML_OPENCL_LINE)
            set(GGML_HAS_OPENCL TRUE)
            message(STATUS "GGML has OpenCL support")
        endif()
        
        # Check for CUDA support
        file(STRINGS "${GGML_HEADER_FILE}" GGML_CUDA_LINE REGEX "GGML_USE_CUDA")
        if(GGML_CUDA_LINE)
            set(GGML_HAS_CUDA TRUE)
            message(STATUS "GGML has CUDA support")
        endif()
    endif()
    
    # Feature detection - Vulkan support
    if(EXISTS "${GGML_INCLUDE_DIR}/ggml-vulkan.h")
        set(GGML_HAS_VULKAN TRUE)
        message(STATUS "GGML has Vulkan support")
    endif()

    # Feature detection - Metal support
    if(EXISTS "${GGML_INCLUDE_DIR}/ggml-metal.h")
        set(GGML_HAS_METAL TRUE)
        message(STATUS "GGML has Metal support")
    endif()
endif()
