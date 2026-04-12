cmake_minimum_required(VERSION 3.22)

# ── Project ───────────────────────────────────────────────────────────────────
# CUDA is listed as an optional language; enable_language(CUDA) is called below
# only when CA_CUDA=ON so that non-CUDA builds don't require nvcc at all.
project(CollAlgo C)

# ── Options ───────────────────────────────────────────────────────────────────
option(CA_CUDA    "Build with CUDA support"   OFF)
option(CA_PROFILE "Build with CB_PROFILE set" OFF)

# ── C standard ────────────────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── compile_commands.json (for clangd) ───────────────────────────────────────
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── MPI ───────────────────────────────────────────────────────────────────────
find_package(MPI REQUIRED C)

# ── CUDA ──────────────────────────────────────────────────────────────────────
if(CA_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Allow device code in static libs to be linked later (separable compilation)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# ── Vendor: CollBench (built separately via `make vendor`) ───────────────────
add_library(collbench STATIC IMPORTED)
set_target_properties(collbench PROPERTIES
    IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/CollBench/build/lib/libcollbench.a
    INTERFACE_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/CollBench/include
)

# ── Common compile options ────────────────────────────────────────────────────
set(WARN_OPTIONS -Wall -Wextra -Wpedantic)

# ── collalgo library ──────────────────────────────────────────────────────────
file(GLOB LIB_C_SRCS CONFIGURE_DEPENDS src/*.c)

if(CA_CUDA)
    file(GLOB LIB_CUDA_SRCS CONFIGURE_DEPENDS src/cuda/*.cu)
endif()

add_library(collalgo STATIC ${LIB_C_SRCS} ${LIB_CUDA_SRCS})

target_include_directories(collalgo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor/CollBench/include>
)

target_compile_options(collalgo PRIVATE
    $<$<COMPILE_LANGUAGE:C>:${WARN_OPTIONS}>
)

target_compile_definitions(collalgo PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    $<$<BOOL:${CA_CUDA}>:CA_CUDA>
    $<$<BOOL:${CA_PROFILE}>:CB_PROFILE>
)

# native tuning only for Release/RelWithDebInfo
target_compile_options(collalgo PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:C>,$<NOT:$<CONFIG:Debug>>>:-march=native>
)

if(CA_CUDA)
    target_link_libraries(collalgo PUBLIC CUDA::cudart)
    # Expose CUDA C++ runtime so consumers don't have to spell it out
    set_target_properties(collalgo PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

target_link_libraries(collalgo PUBLIC MPI::MPI_C)

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

target_compile_definitions(bench PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    $<$<BOOL:${CA_CUDA}>:CA_CUDA>
    $<$<BOOL:${CA_PROFILE}>:CB_PROFILE>
)

target_link_libraries(bench PRIVATE
    collalgo
    collbench
    m
)

# Place the binary under build/bin/ to mirror the old layout
set_target_properties(bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
