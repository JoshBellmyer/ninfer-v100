cmake_minimum_required(VERSION 3.28)

# Upstream NInfer is compiled only for sm_120a. This port adds sm_70 (Volta /
# Tesla V100) as a peer compile-time target. Validate before CUDA compiler detection so CMake never
# injects a generic host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a" AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a or 70 (Volta port); got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

# __CUDA_ARCH__ is only ever defined during device-code compilation, so it can't guide
# host-side dispatch decisions (e.g. "call the SIMT kernel instead of the mma one on
# Volta") — those live in plain .cpp/.cu host functions. This is the equivalent signal
# for host code.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  add_compile_definitions(NINFER_VOLTA_BUILD=1)
endif()

project(ninfer LANGUAGES C CXX CUDA)

# CUTLASS: Volta (sm_70) has real FP16 tensor cores (mma.sync.m8n8k4) but no bf16/int
# tensor-core support and no equivalent to the mma.cuh helpers this codebase's Ampere+
# kernels use. Rather than hand-writing raw Volta mma.sync PTX (a large, error-prone
# undertaking), the Volta build vendors NVIDIA's own CUTLASS
# GEMM template library, which has first-class, NVIDIA-maintained Sm70 support via its
# legacy ("2.x-style") template API. Header-only; fetched only for the Volta build since
# sm_120a already has its own tensor-core kernels. Pinned to the same v4.4.2 tag
# independently verified (via a standalone sanity GEMM, correctness-checked against
# CUTLASS's own reference kernel on real V100 hardware) to build and run correctly here.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  include(FetchContent)
  set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
  set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Disable CUTLASS tests")
  set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Disable CUTLASS examples")
  FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/nvidia/cutlass.git
    GIT_TAG v4.4.2
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(cutlass)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The target runtime headers instantiate very large template TUs; MSVC's default
# per-section object limit is far too small for them. /utf-8 satisfies the bundled fmt
# unicode static assert and keeps char literals on one encoding.
if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# Vendored as a static library: suppress the dllimport export/import decoration.
if(WIN32)
  add_compile_definitions(UTF8PROC_STATIC)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

# sm_120a needs CUDA 13.1+; the sm_70 Volta port targets CUDA 12.8, the last
# toolkit that still assembles sm_70 (CUDA 13.x dropped it). Each arch enforces
# its own floor rather than a single global one.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "sm_120a requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "The sm_70 Volta port requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  message(FATAL_ERROR
    "sm_70 requires CUDA 12.8; CUDA 13.x no longer assembles Volta device code (found "
    "${CMAKE_CUDA_COMPILER_VERSION})")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

# Link the shared CUDA runtime into every executable. With CMAKE_CUDA_RUNTIME_LIBRARY
# unset, CMake appends the static pair (cudadevrt;cudart_static) to each Windows link line,
# which collides with the shared cudart already linked through CUDA::cudart (LNK2005).
if(WIN32)
  set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# --- Media dependencies (FFmpeg + libcurl) ------------------------------------
# Linux resolves both through pkg-config. Windows has no pkg-config; prebuilt shared
# builds are expected under a local prefix (see README "Quick start"):
#   <prefix>/ffmpeg/{include,lib,bin}  e.g. BtbN FFmpeg-Builds win64 gpl-shared
#   <prefix>/curl/{include,lib,bin}    e.g. curl/curl win64-share release zip
# The imported targets keep the pkg-config names so target link lines stay portable.
if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  # One flat output directory keeps the staged runtime DLLs next to every executable.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

  if(DEFINED ENV{NINFER_MEDIA_PREFIX})
    file(TO_CMAKE_PATH "$ENV{NINFER_MEDIA_PREFIX}" ninfer_media_prefix_default)
  else()
    set(ninfer_media_prefix_default "C:/ninfer-deps")
  endif()
  set(NINFER_FFMPEG_ROOT
      "${ninfer_media_prefix_default}/ffmpeg"
      CACHE PATH "Prebuilt Windows FFmpeg root containing include/, lib/, bin/")
  set(NINFER_CURL_ROOT
      "${ninfer_media_prefix_default}/curl"
      CACHE PATH "Prebuilt Windows curl root containing include/, lib/, bin/")

  # --- FFmpeg -----------------------------------------------------------------
  find_path(NINFER_FFMPEG_INCLUDE_DIR NAMES libavformat/avformat.h
            HINTS ${NINFER_FFMPEG_ROOT}/include NO_DEFAULT_PATH)
  foreach(ninfer_ffmpeg_lib IN ITEMS avformat avcodec avutil swscale)
    string(TOUPPER "${ninfer_ffmpeg_lib}" ninfer_ffmpeg_lib_upper)
    find_library(NINFER_FFMPEG_${ninfer_ffmpeg_lib_upper}_LIBRARY NAMES ${ninfer_ffmpeg_lib}
                 PATHS ${NINFER_FFMPEG_ROOT}/lib NO_DEFAULT_PATH)
  endforeach()
  if(NOT NINFER_FFMPEG_INCLUDE_DIR
     OR NOT NINFER_FFMPEG_AVFORMAT_LIBRARY
     OR NOT NINFER_FFMPEG_AVCODEC_LIBRARY
     OR NOT NINFER_FFMPEG_AVUTIL_LIBRARY
     OR NOT NINFER_FFMPEG_SWSCALE_LIBRARY)
    message(FATAL_ERROR
      "Windows FFmpeg not found under '${NINFER_FFMPEG_ROOT}'. Install a prebuilt win64 "
      "shared build (libavformat >= 60, libavcodec >= 60, libavutil >= 58, libswscale >= 7) "
      "and set -DNINFER_FFMPEG_ROOT to its root (include/, lib/, bin/).")
  endif()

  # Guard against an FFmpeg older than the API floor used by media/decode. The .pc files
  # carry the library version on every prebuilt layout (the LIB*_VERSION_MAJOR macros were
  # removed from the headers in FFmpeg 8).
  function(ninfer_check_ffmpeg_version component minimum)
    string(TOLOWER "lib${component}.pc" ninfer_ffmpeg_pc_file)
    set(ninfer_ffmpeg_pc_path "${NINFER_FFMPEG_ROOT}/lib/pkgconfig/${ninfer_ffmpeg_pc_file}")
    if(NOT EXISTS "${ninfer_ffmpeg_pc_path}")
      message(FATAL_ERROR "Missing FFmpeg pkg-config file: ${ninfer_ffmpeg_pc_path}")
    endif()
    file(STRINGS "${ninfer_ffmpeg_pc_path}" ninfer_ffmpeg_version_line
         REGEX "^Version:[ \t]*[0-9]+\.[0-9]+\.[0-9]+")
    if(NOT ninfer_ffmpeg_version_line)
      message(FATAL_ERROR "Could not read Version from ${ninfer_ffmpeg_pc_path}")
    endif()
    string(REGEX REPLACE ".*Version:[ \t]*([0-9]+)\.[0-9]+\.[0-9]+.*" "\\1"
           ninfer_ffmpeg_version_major "${ninfer_ffmpeg_version_line}")
    if(ninfer_ffmpeg_version_major VERSION_LESS ${minimum})
      message(FATAL_ERROR
        "FFmpeg ${component} major version ${ninfer_ffmpeg_version_major} is below the "
        "required minimum ${minimum}")
    endif()
  endfunction()
  ninfer_check_ffmpeg_version(avformat 60)
  ninfer_check_ffmpeg_version(avcodec 60)
  ninfer_check_ffmpeg_version(avutil 58)
  ninfer_check_ffmpeg_version(swscale 7)

  add_library(PkgConfig::FFMPEG INTERFACE IMPORTED)
  set_target_properties(PkgConfig::FFMPEG PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NINFER_FFMPEG_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES
      "${NINFER_FFMPEG_AVFORMAT_LIBRARY};${NINFER_FFMPEG_AVCODEC_LIBRARY};${NINFER_FFMPEG_AVUTIL_LIBRARY};${NINFER_FFMPEG_SWSCALE_LIBRARY}")
  # Stage the runtime DLLs next to every executable at build time (ninfer_stage_runtime_dlls).
  file(GLOB NINFER_FFMPEG_DLLS "${NINFER_FFMPEG_ROOT}/bin/*.dll")
  if(NOT NINFER_FFMPEG_DLLS)
    message(FATAL_ERROR "No FFmpeg runtime DLLs found in '${NINFER_FFMPEG_ROOT}/bin'")
  endif()
  set(NINFER_RUNTIME_DLLS ${NINFER_FFMPEG_DLLS})

  # --- libcurl ------------------------------------------------------------------
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    find_path(NINFER_CURL_INCLUDE_DIR NAMES curl/curl.h
              HINTS ${NINFER_CURL_ROOT}/include NO_DEFAULT_PATH)
    find_library(NINFER_CURL_LIBRARY NAMES libcurl curl
                 PATHS ${NINFER_CURL_ROOT}/lib/x64 ${NINFER_CURL_ROOT}/lib NO_DEFAULT_PATH)
    if(NOT NINFER_CURL_INCLUDE_DIR OR NOT NINFER_CURL_LIBRARY)
      message(FATAL_ERROR
        "Windows libcurl not found under '${NINFER_CURL_ROOT}'. Install a prebuilt win64 "
        "shared build (>= 7.85) and set -DNINFER_CURL_ROOT to its root (include/, lib/, bin/).")
    endif()
    add_library(PkgConfig::LIBCURL INTERFACE IMPORTED)
    set_target_properties(PkgConfig::LIBCURL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NINFER_CURL_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${NINFER_CURL_LIBRARY}")
    file(GLOB NINFER_CURL_DLLS "${NINFER_CURL_ROOT}/bin/*.dll")
    if(NOT NINFER_CURL_DLLS)
      message(FATAL_ERROR "No curl runtime DLLs found in '${NINFER_CURL_ROOT}/bin'")
    endif()
    list(APPEND NINFER_RUNTIME_DLLS ${NINFER_CURL_DLLS})
  endif()

  # Multi-config generators (Visual Studio) place executables under bin/<Config>/; stage the DLLs
  # there at build time so every executable can load them.
  if(CMAKE_CONFIGURATION_TYPES)
    set(NINFER_RUNTIME_DLL_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
  else()
    set(NINFER_RUNTIME_DLL_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  endif()
  add_custom_target(ninfer_stage_runtime_dlls ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${NINFER_RUNTIME_DLL_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NINFER_RUNTIME_DLLS} "${NINFER_RUNTIME_DLL_DIR}"
    COMMENT "Staging FFmpeg/curl runtime DLLs next to the executables"
    VERBATIM)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
endif()

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  # Product operational logging is an offline, repository-pinned dependency. Core-only builds do
  # not configure or link it.
  add_subdirectory(third_party/spdlog)
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
