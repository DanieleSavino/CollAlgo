# CollAlgo

> [!WARNING]
> This README was AI-generated from the source code and headers. Some details may be imprecise, incomplete, or outdated. When in doubt, refer to the headers in `include/CollAlgo/` as the authoritative reference.

**CollAlgo** is a C library implementing custom MPI collective communication algorithms, developed as part of a Bachelor's thesis in Computer Science at **Sapienza Università di Roma**. It provides optimized implementations of gather, scatter, broadcast, and reduce collectives based on the BiNE (Binary Negabinary Exchange) algorithm, with optional CUDA-aware execution and built-in profiling support via [CollBench](vendor/CollBench/README.MD).

---

## Overview

Standard MPI collectives are highly optimized for common hardware topologies, but research into alternative algorithms requires clean, instrumented implementations to measure and compare performance. CollAlgo provides BiNE-based implementations of `gather`, `gatherv`, `scatter`, `scatterv`, `bcast`, and `reduce`, ready to be benchmarked with nanosecond-resolution timing through CollBench. Collectives automatically detect whether buffers reside on the host or device and dispatch accordingly — no changes to call sites required.

---

## Features

- **BiNE collective algorithms**: gather, gatherv, scatter, scatterv, bcast, reduce
- **CUDA-aware execution**: collectives detect device pointers via `CA_is_devptr()` and dispatch to GPU kernels automatically; falls back to host path when CUDA is disabled or buffers are on the host
- **Profiling integration**: instrument with CollBench by building with `-DCA_PROFILE=ON`; zero overhead otherwise
- **Correctness benchmarks**: built-in bench executable verifies output against expected results
- **CMake build system**: single configure step handles vendor, library, and bench; `compile_commands.json` exported automatically for LSP support

---

## Requirements

- An MPI implementation (e.g. Open MPI, MPICH)
- CMake ≥ 3.22
- C11-compatible compiler
- *(Optional)* CUDA Toolkit ≥ 12 for GPU-aware collectives
- *(Optional)* [`bear`](https://github.com/rizsotto/Bear) for regenerating `compile_commands.json`

---

## Building

```sh
# Configure (CPU only)
cmake -B build

# Configure with CUDA support
cmake -B build -DCA_CUDA=ON

# Configure with profiling enabled (bakes CB_PROFILE into collalgo_profile and bench)
cmake -B build -DCA_PROFILE=ON

# Configure with both
cmake -B build -DCA_CUDA=ON -DCA_PROFILE=ON

# Build everything (vendor/CollBench is built automatically if missing)
cmake --build build --parallel

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

Build outputs:
```
build/
├── bin/
│   └── bench                    # Benchmark executable
├── libcollalgo.a                # Release library
└── libcollalgo_profile.a        # Profiling library (CB_PROFILE baked in)

vendor/CollBench/build/lib/
└── libcollbench.a               # Built automatically on first cmake --build
```

---

## Project Structure

```
CollAlgo/
├── include/CollAlgo/
│   ├── bcast.h             # Broadcast interface
│   ├── bine.h              # BiNE utilities (negabinary conversion, log2, etc.)
│   ├── gather.h            # Gather / gatherv interface
│   ├── reduce.h            # Reduce interface
│   ├── scatter.h           # Scatter / scatterv interface
│   ├── utils.h             # Utilities, error macros, unified host/device alloc
│   └── cuda/
│       ├── reduce.h        # CUDA reduce kernel interface
│       └── utils.h         # CA_is_devptr and CUDA helpers
├── src/
│   ├── bcast.c
│   ├── gather.c
│   ├── reduce.c
│   ├── scatter.c
│   └── cuda/
│       ├── reduce.cu       # GPU reduce kernel
│       └── utils.cu        # _CA_cuda_is_devptr implementation
├── bench/
│   ├── include/bench/
│   │   ├── allgather.h
│   │   ├── alltoall.h
│   │   ├── bcast.h
│   │   ├── gather.h
│   │   ├── reduce.h
│   │   ├── scatter.h
│   │   └── cuda/
│   │       └── reduce.h
│   └── src/
│       ├── main.c
│       ├── allgather.c
│       ├── alltoall.c
│       ├── bcast.c
│       ├── gather.c
│       ├── reduce.c
│       ├── scatter.c
│       └── cuda/
│           └── reduce.cu
├── vendor/
│   └── CollBench/          # Profiling and timing library (built by CMake)
├── scripts/
│   ├── tree.py             # Tree topology visualizer
│   ├── butterfly.py        # Butterfly topology visualizer
│   ├── profile_np8_leonardo.sbatch
│   ├── profile_np8_meluxina.sbatch
│   └── profile_np8_mn5.sbatch
└── CMakeLists.txt
```

---

## API

Include the relevant header for the collective you want to use:

```c
#include "CollAlgo/gather.h"
#include "CollAlgo/scatter.h"
#include "CollAlgo/bcast.h"
#include "CollAlgo/reduce.h"
```

### Gather

```c
int CA_bine_gather(const void *sendbuff, int sendcount, MPI_Datatype sendtype,
                   void *recvbuff, int recvcount, MPI_Datatype recvtype,
                   int root, MPI_Comm comm);

int CA_bine_gatherv(const void *sendbuff, int sendcount, MPI_Datatype sendtype,
                    void *recvbuff, int *recvcounts, const int *displs, MPI_Datatype recvtype,
                    int root, MPI_Comm comm);
```

### Scatter

```c
int CA_bine_scatter(const void *sendbuff, int sendcount, MPI_Datatype sendtype,
                    void *recvbuff, int recvcount, MPI_Datatype recvtype,
                    int root, MPI_Comm comm);

int CA_bine_scatterv(const void *sendbuff, int *sendcounts, const int *displs, MPI_Datatype sendtype,
                     void *recvbuff, int recvcount, MPI_Datatype recvtype,
                     int root, MPI_Comm comm);
```

### Reduce

```c
int CA_bine_reduce(const void *sendbuff, void *recvbuff, int count,
                   MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
```

All functions return `MPI_SUCCESS` on success, or an MPI error code on failure. All collective functions require the communicator size to be a power of two.

### CUDA-aware usage

Collectives automatically detect whether buffers are on the device using `CA_is_devptr()`. Simply pass device pointers allocated with `cudaMalloc` — no API changes needed:

```c
int *d_send, *d_recv;
cudaMalloc(&d_send, count * sizeof(int));
cudaMalloc(&d_recv, count * sizeof(int));

// Automatically dispatches to GPU path
CA_bine_reduce(d_send, d_recv, count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
```

Requires building with `-DCA_CUDA=ON`. When built without CUDA, `CA_is_devptr()` always returns 0 and all unified macros (`CA_UMALLOC`, `CA_UFREE`, etc.) collapse to their host equivalents with zero overhead.

---

## Profiling

CollAlgo integrates with [CollBench](vendor/CollBench/README.MD) for per-operation nanosecond-resolution timing. Build with `-DCA_PROFILE=ON` to enable it:

```sh
cmake -B build -DCA_PROFILE=ON -DCA_CUDA=ON
cmake --build build --parallel
srun -n 8 ./build/bin/bench
```

Profiling output is written as JSON under `out/`:

```
out/
├── tree/
│   ├── bine_bcast_dhlv.json
│   ├── bine_gather.json
│   ├── bine_gatherv.json
│   ├── bine_reduce.json
│   └── bine_reduce_cuda.json    # Written when sendbuff is a device pointer
└── butterfly/
    ├── bine_allgather.json
    └── bine_alltoall.json
```

Omit `-DCA_PROFILE=ON` for a production build — all instrumentation compiles away with zero overhead.

---

## HPC Scripts

SLURM job scripts for profiling runs are provided under `scripts/`. Each script builds, runs the benchmark, and generates tree and butterfly plots.

| Script | System | Partition |
|--------|--------|-----------|
| `profile_np8_leonardo.sbatch` | Leonardo (CINECA) | `boost_usr_prod` |
| `profile_np8_meluxina.sbatch` | MeluXina (LuxProvide) | `gpu` |
| `profile_np8_mn5.sbatch` | MareNostrum 5 (BSC) | `acc_debug` / `acc_ehpc` |

All scripts use 4 nodes × 2 GPUs = 8 MPI ranks.

---

## License

MIT License

Copyright (c) 2025 Daniele Savino — Sapienza Università di Roma

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
