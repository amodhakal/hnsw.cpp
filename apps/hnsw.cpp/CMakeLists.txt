cmake_minimum_required(VERSION 3.18)

project(hnsw_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(hnsw_core STATIC src/hnsw.cpp)
target_include_directories(
    hnsw_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(hnsw_core PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(hnsw_core PRIVATE -O3 -Wall -Wextra)
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
