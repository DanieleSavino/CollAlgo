cmake_minimum_required(VERSION 3.22)
project(CollAlgo C CUDA)
option(CA_CUDA    "Build with CUDA support"   OFF)
option(CA_PROFILE "Build with CB_PROFILE set" OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(MPI REQUIRED C)
if(CA_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# ── Vendor: build CollBench if the static lib is missing ─────────────────────
set(COLLBENCH_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/CollBench)
set(COLLBENCH_LIB      ${COLLBENCH_SRC_DIR}/build/lib/libcollbench.a)

# Map CMake build type → CollBench Makefile BUILD= flag
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(COLLBENCH_BUILD_FLAG BUILD=debug)
else()
    set(COLLBENCH_BUILD_FLAG BUILD=release)
endif()

add_custom_command(
    OUTPUT  ${COLLBENCH_LIB}
    COMMAND ${CMAKE_MAKE_PROGRAM} ${COLLBENCH_BUILD_FLAG}
    WORKING_DIRECTORY ${COLLBENCH_SRC_DIR}
    COMMENT "Building vendor/CollBench (${COLLBENCH_BUILD_FLAG})"
    VERBATIM
)

add_custom_target(collbench_vendor
    DEPENDS ${COLLBENCH_LIB}
)

add_library(collbench STATIC IMPORTED GLOBAL)
add_dependencies(collbench collbench_vendor)
set_target_properties(collbench PROPERTIES
    IMPORTED_LOCATION             ${COLLBENCH_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${COLLBENCH_SRC_DIR}/include
)

set(WARN_OPTIONS -Wall -Wextra -Wpedantic)

# ── Helper: populate a collalgo-variant target ────────────────────────────────
function(configure_collalgo_target target profile)
    file(GLOB LIB_C_SRCS CONFIGURE_DEPENDS src/*.c)
    if(CA_CUDA)
        file(GLOB LIB_CUDA_SRCS CONFIGURE_DEPENDS src/cuda/*.cu)
    endif()
    add_library(${target} STATIC ${LIB_C_SRCS} ${LIB_CUDA_SRCS})
    target_include_directories(${target}
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE $<BUILD_INTERFACE:${COLLBENCH_SRC_DIR}/include>
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${WARN_OPTIONS}>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<NOT:$<CONFIG:Debug>>>:-march=native>
    )
    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
        $<$<BOOL:${CA_CUDA}>:CA_CUDA>
        $<$<BOOL:${profile}>:CB_PROFILE>
    )
    if(CA_CUDA)
        target_link_libraries(${target} PUBLIC CUDA::cudart)
        set_target_properties(${target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=128>)
    endif()
    target_link_libraries(${target} PUBLIC MPI::MPI_C)
endfunction()

# ── Library targets ───────────────────────────────────────────────────────────
configure_collalgo_target(collalgo         OFF)
configure_collalgo_target(collalgo_profile ON)

# ── bench executable ──────────────────────────────────────────────────────────
file(GLOB BENCH_SRCS CONFIGURE_DEPENDS bench/src/*.c)
if(CA_CUDA)
    file(GLOB BENCH_CUDA_SRCS CONFIGURE_DEPENDS bench/src/cuda/*.cu)
endif()
add_executable(bench ${BENCH_SRCS} ${BENCH_CUDA_SRCS})
target_include_directories(bench PRIVATE bench/include)
target_compile_options(bench PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${WARN_OPTIONS}>
    $<$<NOT:$<CONFIG:Debug>>:-march=native>
)
if(CA_CUDA)
    target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=128>)
endif()
target_compile_definitions(bench PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    $<$<BOOL:${CA_CUDA}>:CA_CUDA>
    $<$<BOOL:${CA_PROFILE}>:CB_PROFILE>
)
if(CA_PROFILE)
    set(_collalgo_link collalgo_profile)
else()
    set(_collalgo_link collalgo)
endif()
target_link_libraries(bench PRIVATE
    ${_collalgo_link}
    collbench
    m
)
set_target_properties(bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
