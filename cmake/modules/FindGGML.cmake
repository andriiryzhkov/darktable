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

# First try to use the CMake config files if they exist
find_package(ggml CONFIG QUIET)

if(ggml_FOUND)
    set(GGML_FOUND TRUE)
    
    # Variables are already set by the CMake config file
    # We just need to set our custom variables
    set(GGML_LIBRARIES ggml::ggml)
    
    # Get include directories from target if available
    get_target_property(GGML_INCLUDE_DIRS ggml::ggml INTERFACE_INCLUDE_DIRECTORIES)
    
    # Try to extract version info from the config files
    if(TARGET ggml::ggml)
        if(NOT GGML_VERSION)
            get_target_property(GGML_VERSION ggml::ggml VERSION)
        endif()
    endif()
else()
    # If CMake config files not found, try to find libraries and headers manually
    
    # Use pkg-config to get hints about paths
    libfind_pkg_check_modules(GGML_PKGCONF ggml)
    
    # Define search names for the libraries
    set(GGML_NAMES ${GGML_NAMES} ggml libggml)
    
    # Common paths
    set(COMMON_INCLUDE_PATHS
        ${GGML_PKGCONF_INCLUDE_DIRS}
        /usr/include
        /usr/local/include
        /opt/local/include
        /opt/homebrew/include
        $ENV{MINGW_PREFIX}/include
        /mingw64/include
        /ucrt64/include
        /clang64/include
        C:/msys64/mingw64/include
        C:/msys64/ucrt64/include
        C:/msys64/clang64/include
    )
    
    set(COMMON_LIBRARY_PATHS
        ${GGML_PKGCONF_LIBRARY_DIRS}
        /usr/lib
        /usr/local/lib
        /usr/lib64
        /usr/local/lib64
        /opt/local/lib
        /opt/homebrew/lib
        $ENV{MINGW_PREFIX}/lib
        /mingw64/lib
        /ucrt64/lib
        /clang64/lib
        C:/msys64/mingw64/lib
        C:/msys64/ucrt64/lib
        C:/msys64/clang64/lib
    )
    
    # Look for GGML header
    find_path(GGML_INCLUDE_DIR 
        NAMES ggml.h
        PATHS ${COMMON_INCLUDE_PATHS}
        PATH_SUFFIXES ggml
    )
    mark_as_advanced(GGML_INCLUDE_DIR)
    
    # Find the core GGML library
    find_library(GGML_LIBRARY 
        NAMES ${GGML_NAMES}
        PATHS ${COMMON_LIBRARY_PATHS}
    )
    mark_as_advanced(GGML_LIBRARY)
    
    # Find the GGML components
    find_library(GGML_BASE_LIBRARY 
        NAMES ggml-base libggml-base
        PATHS ${COMMON_LIBRARY_PATHS}
    )
    mark_as_advanced(GGML_BASE_LIBRARY)

    find_library(GGML_BLAS_LIBRARY 
        NAMES ggml-blas libggml-blas
        PATHS ${COMMON_LIBRARY_PATHS}
    )
    mark_as_advanced(GGML_BLAS_LIBRARY)
    
    find_library(GGML_CPU_LIBRARY 
        NAMES ggml-cpu libggml-cpu
        PATHS ${COMMON_LIBRARY_PATHS}
    )
    mark_as_advanced(GGML_CPU_LIBRARY)
    
    # Find optional components
    find_library(GGML_OPENCL_LIBRARY 
        NAMES ggml-opencl libggml-opencl
        PATHS ${COMMON_LIBRARY_PATHS}
    )
    mark_as_advanced(GGML_OPENCL_LIBRARY)
    
    find_library(GGML_VULKAN_LIBRARY 
        NAMES ggml-vulkan libggml-vulkan
        PATHS ${COMMON_LIBRARY_PATHS}
    )
    mark_as_advanced(GGML_VULKAN_LIBRARY)

    find_library(GGML_RPC_LIBRARY 
        NAMES ggml-rpc libggml-rpc
        PATHS ${COMMON_LIBRARY_PATHS}
    )
    mark_as_advanced(GGML_RPC_LIBRARY)
    
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
    
    # Check for necessary dependencies
    find_package(Threads QUIET)
    
    # Platform-specific dependencies
    set(GGML_EXTRA_DEPS "")
    if(WIN32)
        # Nothing specific for Windows
    elseif(APPLE)
        # macOS specific dependencies
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            list(APPEND GGML_EXTRA_DEPS ${ACCELERATE_FRAMEWORK})
        endif()
    else()
        # Linux specific dependencies
        find_library(MATH_LIBRARY m)
        if(MATH_LIBRARY)
            list(APPEND GGML_EXTRA_DEPS ${MATH_LIBRARY})
        endif()
    endif()
    
    # Add thread libraries if found
    if(Threads_FOUND)
        list(APPEND GGML_EXTRA_DEPS ${CMAKE_THREAD_LIBS_INIT})
    endif()
    
    # Determine version from headers
    if(GGML_INCLUDE_DIR)
        set(GGML_HEADER_FILE "${GGML_INCLUDE_DIR}/ggml.h")
        if(NOT EXISTS "${GGML_HEADER_FILE}" AND EXISTS "${GGML_INCLUDE_DIR}/ggml/ggml.h")
            set(GGML_HEADER_FILE "${GGML_INCLUDE_DIR}/ggml/ggml.h")
        endif()
        
        if(EXISTS "${GGML_HEADER_FILE}")
            # Try multiple version definition patterns
            file(STRINGS "${GGML_HEADER_FILE}" GGML_VERSION_LINE REGEX "^#define[ \t]+(GGML_VERSION|GGML_VERSION_STRING|GGML_BUILD_NUMBER)[ \t]+\"?[0-9.]+\"?")
            if(GGML_VERSION_LINE)
                string(REGEX REPLACE "^#define[ \t]+(GGML_VERSION|GGML_VERSION_STRING|GGML_BUILD_NUMBER)[ \t]+\"?([0-9.]+)\"?.*" "\\2" GGML_VERSION "${GGML_VERSION_LINE}")
            else()
                # Try to find version in component format
                file(STRINGS "${GGML_HEADER_FILE}" GGML_VERSION_MAJOR_LINE REGEX "^#define[ \t]+GGML_VERSION_MAJOR[ \t]+[0-9]+")
                file(STRINGS "${GGML_HEADER_FILE}" GGML_VERSION_MINOR_LINE REGEX "^#define[ \t]+GGML_VERSION_MINOR[ \t]+[0-9]+")
                
                if(GGML_VERSION_MAJOR_LINE AND GGML_VERSION_MINOR_LINE)
                    string(REGEX REPLACE "^#define[ \t]+GGML_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" GGML_VERSION_MAJOR "${GGML_VERSION_MAJOR_LINE}")
                    string(REGEX REPLACE "^#define[ \t]+GGML_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" GGML_VERSION_MINOR "${GGML_VERSION_MINOR_LINE}")
                    
                    set(GGML_VERSION "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}")
                    
                    # Check for patch version
                    file(STRINGS "${GGML_HEADER_FILE}" GGML_VERSION_PATCH_LINE REGEX "^#define[ \t]+GGML_VERSION_PATCH[ \t]+[0-9]+")
                    if(GGML_VERSION_PATCH_LINE)
                        string(REGEX REPLACE "^#define[ \t]+GGML_VERSION_PATCH[ \t]+([0-9]+).*" "\\1" GGML_VERSION_PATCH "${GGML_VERSION_PATCH_LINE}")
                        set(GGML_VERSION "${GGML_VERSION}.${GGML_VERSION_PATCH}")
                    endif()
                endif()
            endif()
        endif()
    endif()
    
    # Apply standard rules to see if required components are found
    include(FindPackageHandleStandardArgs)
    
    # GGML requirements
    find_package_handle_standard_args(GGML
        REQUIRED_VARS GGML_LIBRARY GGML_INCLUDE_DIR
        VERSION_VAR GGML_VERSION
    )
    
    # Set libraries and include directories if GGML is found
    if(GGML_FOUND)
        set(GGML_LIBRARIES ${GGML_LIBRARY} ${GGML_COMPONENT_LIBRARIES} ${GGML_EXTRA_DEPS})
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
    endif()
endif()

# Show status messages
if(GGML_FOUND AND NOT GGML_FIND_QUIETLY)
    message(STATUS "Found GGML: ${GGML_LIBRARIES}")
    if(GGML_VERSION)
        message(STATUS "GGML version: ${GGML_VERSION}")
    endif()
endif()

# Verify requested version
if(GGML_FOUND AND GGML_FIND_VERSION AND GGML_VERSION)
    set(GGML_FAILED_VERSION_CHECK true)
    
    if(GGML_FIND_VERSION_EXACT)
        if(GGML_VERSION VERSION_EQUAL GGML_FIND_VERSION)
            set(GGML_FAILED_VERSION_CHECK false)
        endif()
    else()
        if(GGML_VERSION VERSION_EQUAL GGML_FIND_VERSION OR GGML_VERSION VERSION_GREATER GGML_FIND_VERSION)
            set(GGML_FAILED_VERSION_CHECK false)
        endif()
    endif()
    
    if(GGML_FAILED_VERSION_CHECK)
        if(GGML_FIND_REQUIRED AND NOT GGML_FIND_QUIETLY)
            if(GGML_FIND_VERSION_EXACT)
                message(FATAL_ERROR "GGML version check failed. Version ${GGML_VERSION} was found, version ${GGML_FIND_VERSION} is needed exactly.")
            else()
                message(FATAL_ERROR "GGML version check failed. Version ${GGML_VERSION} was found, at least version ${GGML_FIND_VERSION} is required")
            endif()
        endif()
        
        set(GGML_FOUND false)
    endif()
endif()

# Installation helper function
function(install_ggml_libraries COMPONENT DESTINATION)
    # Install core library
    if(GGML_LIBRARY)
        get_filename_component(GGML_LIBRARY_NAME "${GGML_LIBRARY}" NAME)
        install(FILES "${GGML_LIBRARY}" 
                DESTINATION "${DESTINATION}"
                COMPONENT "${COMPONENT}")
        
        # On Windows, also install DLL if we're using .dll.a import library
        if(WIN32 AND GGML_LIBRARY_NAME MATCHES "\\.dll\\.a$")
            string(REGEX REPLACE "\\.dll\\.a$" ".dll" GGML_DLL_NAME "${GGML_LIBRARY_NAME}")
            get_filename_component(GGML_LIBRARY_DIR "${GGML_LIBRARY}" DIRECTORY)
            get_filename_component(GGML_BIN_DIR "${GGML_LIBRARY_DIR}/../bin" ABSOLUTE)
            if(EXISTS "${GGML_BIN_DIR}/${GGML_DLL_NAME}")
                install(FILES "${GGML_BIN_DIR}/${GGML_DLL_NAME}" 
                        DESTINATION "${DESTINATION}"
                        COMPONENT "${COMPONENT}")
            endif()
        endif()
    endif()
    
    # Install all component libraries
    foreach(LIB_VAR GGML_BASE_LIBRARY GGML_BLAS_LIBRARY GGML_CPU_LIBRARY GGML_OPENCL_LIBRARY GGML_VULKAN_LIBRARY GGML_RPC_LIBRARY)
        if(${LIB_VAR})
            get_filename_component(LIB_NAME "${${LIB_VAR}}" NAME)
            install(FILES "${${LIB_VAR}}" 
                    DESTINATION "${DESTINATION}"
                    COMPONENT "${COMPONENT}")
            
            # On Windows, also install DLL if we're using .dll.a import library
            if(WIN32 AND LIB_NAME MATCHES "\\.dll\\.a$")
                string(REGEX REPLACE "\\.dll\\.a$" ".dll" DLL_NAME "${LIB_NAME}")
                get_filename_component(LIB_DIR "${${LIB_VAR}}" DIRECTORY)
                get_filename_component(BIN_DIR "${LIB_DIR}/../bin" ABSOLUTE)
                if(EXISTS "${BIN_DIR}/${DLL_NAME}")
                    install(FILES "${BIN_DIR}/${DLL_NAME}" 
                            DESTINATION "${DESTINATION}"
                            COMPONENT "${COMPONENT}")
                endif()
            endif()
        endif()
    endforeach()
endfunction()

# Installation helper function for headers
function(install_ggml_headers COMPONENT DESTINATION)
    if(GGML_INCLUDE_DIR)
        # Install main GGML header
        if(EXISTS "${GGML_INCLUDE_DIR}/ggml.h")
            install(FILES "${GGML_INCLUDE_DIR}/ggml.h" 
                    DESTINATION "${DESTINATION}"
                    COMPONENT "${COMPONENT}")
        endif()
        
        # Install other GGML headers
        file(GLOB GGML_HEADERS "${GGML_INCLUDE_DIR}/ggml-*.h")
        if(GGML_HEADERS)
            install(FILES ${GGML_HEADERS} 
                    DESTINATION "${DESTINATION}"
                    COMPONENT "${COMPONENT}")
        endif()
        
        # Also look for headers in subdirectories
        if(EXISTS "${GGML_INCLUDE_DIR}/ggml/ggml.h")
            file(GLOB GGML_HEADERS "${GGML_INCLUDE_DIR}/ggml/*.h")
            if(GGML_HEADERS)
                install(FILES ${GGML_HEADERS} 
                        DESTINATION "${DESTINATION}"
                        COMPONENT "${COMPONENT}")
            endif()
        endif()
    endif()
endfunction()