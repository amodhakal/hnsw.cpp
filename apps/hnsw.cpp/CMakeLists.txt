cmake_minimum_required(VERSION 3.18)

project(hnsw_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# hnsw_core is a static lib linked into a shared Python module -> needs PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- CPU tuning -------------------------------------------------------------
# HNSW_CPU: "native" (default), "apple-m1", or "none" (portable build).
# Native/M1 builds are NOT portable: don't ship wheels built this way.
set(HNSW_CPU "native" CACHE STRING "CPU target: native | apple-m1 | none")
set_property(CACHE HNSW_CPU PROPERTY STRINGS native apple-m1 none)

option(HNSW_LTO "Enable link-time optimization" ON)

include(CheckCXXCompilerFlag)

set(HNSW_ARCH_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang" AND NOT HNSW_CPU STREQUAL "none")
    # A universal build (arm64 + x86_64) can't take a single -mcpu/-march flag.
    set(_universal FALSE)
    if(APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES ";|x86_64")
        set(_universal TRUE)
    endif()

    if(_universal)
        message(STATUS "hnsw: universal build detected, skipping CPU tuning flags")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
        # ARM: -mcpu. Clang >= 15 and GCC >= 11 support -mcpu=native on arm64.
        set(_cpu_flag "-mcpu=${HNSW_CPU}")
        check_cxx_compiler_flag("${_cpu_flag}" HNSW_HAS_MCPU)
        if(HNSW_HAS_MCPU)
            set(HNSW_ARCH_FLAGS ${_cpu_flag})
        elseif(HNSW_CPU STREQUAL "native")
            # Old toolchain without -mcpu=native: fall back to the M1 target.
            check_cxx_compiler_flag("-mcpu=apple-m1" HNSW_HAS_MCPU_M1)
            if(HNSW_HAS_MCPU_M1)
                set(HNSW_ARCH_FLAGS -mcpu=apple-m1)
            endif()
        endif()
    else()
        # x86: -march=native (apple-m1 makes no sense here)
        if(HNSW_CPU STREQUAL "native")
            check_cxx_compiler_flag("-march=native" HNSW_HAS_MARCH)
            if(HNSW_HAS_MARCH)
                set(HNSW_ARCH_FLAGS -march=native)
            endif()
        endif()
    endif()
endif()
message(STATUS "hnsw: arch flags = '${HNSW_ARCH_FLAGS}'")

# ---- Core library -----------------------------------------------------------
add_library(hnsw_core STATIC src/hnsw.cpp)
target_include_directories(
    hnsw_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(hnsw_core PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(hnsw_core PRIVATE -O3 -Wall -Wextra ${HNSW_ARCH_FLAGS})
endif()

if(HNSW_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg LANGUAGES CXX)
    if(_ipo_ok)
        set_property(TARGET hnsw_core PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "hnsw: LTO unsupported: ${_ipo_msg}")
    endif()
endif()

add_executable(hnsw_demo src/main.cpp)
target_link_libraries(hnsw_demo PRIVATE hnsw_core)

enable_testing()
add_executable(hnsw_tests tests/test_hnsw.cpp)
target_link_libraries(hnsw_tests PRIVATE hnsw_core)
add_test(NAME hnsw_tests COMMAND hnsw_tests)

# The Python extension is only built when pybind11 is available. CMake is
# invoked by scikit-build-core in the benchmark project, where it is required.
if(DEFINED SKBUILD)
    find_package(pybind11 CONFIG REQUIRED)
else()
    find_package(pybind11 CONFIG QUIET)
endif()

if(pybind11_FOUND)
    pybind11_add_module(hnsw_cpp src/bindings.cpp)
    target_link_libraries(hnsw_cpp PRIVATE hnsw_core)
    install(TARGETS hnsw_cpp LIBRARY DESTINATION . RUNTIME DESTINATION .)
endif()
