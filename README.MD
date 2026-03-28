# CollAlgo

> [!WARNING]
> This README was AI-generated from the source code and headers. Some details may be imprecise, incomplete, or outdated. When in doubt, refer to the headers in `include/CollAlgo/` as the authoritative reference.

**CollAlgo** is a C library implementing custom MPI collective communication algorithms, developed as part of a Bachelor's thesis in Computer Science at **Sapienza Università di Roma**. It provides optimized implementations of gather, scatter, and broadcast collectives based on the BiNE (Binary Negabinary Exchange) algorithm, with built-in support for profiling via [CollBench](vendor/CollBench/README.MD).

---

## Overview

Standard MPI collectives are highly optimized for common hardware topologies, but research into alternative algorithms requires clean, instrumented implementations to measure and compare performance. CollAlgo provides BiNE-based implementations of `gather`, `gatherv`, `scatter`, `scatterv`, and `bcast`, ready to be benchmarked with nanosecond-resolution timing through CollBench.

---

## Features

- **BiNE collective algorithms**: gather, gatherv, scatter, scatterv, bcast
- **Profiling integration**: instrument with CollBench by compiling with `-DCB_PROFILE`; zero overhead otherwise
- **Correctness benchmarks**: built-in bench executable verifies output against expected results
- **Three build modes**: release, profile, and debug

---

## Requirements

- An MPI implementation (e.g. Open MPI, MPICH)
- C99-compatible compiler accessible as `mpicc`
- GNU Make
- *(Optional)* [`bear`](https://github.com/rizsotto/Bear) for generating `compile_commands.json` for LSP support

---

## Building
```sh
# Build release library (default)
make lib

# Build debug library
make lib-debug

# Build benchmark executable
make bench

# Build benchmark executable with profiling enabled
make bench PROFILE=1

# Build vendor/CollBench
make vendor

# Generate compile_commands.json (requires bear)
make compile_commands.json

# Clean all build artifacts
make clean
```

Build outputs are placed under `build/`:
```
build/
├── lib/
│   ├── libcollalgo.a            # Release library
│   ├── libcollalgo_profile.a    # Profiled library (-DCB_PROFILE)
│   └── libcollalgo_debug.a      # Debug library
└── bin/
    └── bench                    # Benchmark executable
```

---

## Project Structure
```
CollAlgo/
├── include/CollAlgo/
│   ├── bcast.h       # Broadcast interface
│   ├── bine.h        # BiNE utility functions (negabinary conversion, etc.)
│   ├── gather.h      # Gather / gatherv interface
│   ├── scatter.h     # Scatter / scatterv interface
│   └── utils.h       # General utilities
├── src/
│   ├── bcast.c
│   ├── gather.c
│   └── scatter.c
├── bench/
│   ├── include/bench/
│   │   ├── bcast.h
│   │   ├── gather.h
│   │   └── scatter.h
│   └── src/
│       ├── bcast.c
│       ├── gather.c
│       ├── main.c
│       └── scatter.c
├── vendor/
│   └── CollBench/    # Profiling and timing library
├── scripts/
│   └── tree.py -> ../vendor/CollBench/scripts/tree.py
└── Makefile
```

---

## API

Include the relevant header for the collective you want to use:
```c
#include "CollAlgo/gather.h"
#include "CollAlgo/scatter.h"
#include "CollAlgo/bcast.h"
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

All functions return `MPI_SUCCESS` on success, or an MPI error code on failure. All collective functions require the communicator size to be a power of two.

---

## Profiling

CollAlgo integrates with [CollBench](vendor/CollBench/README.MD) for per-operation nanosecond-resolution timing. Build with `PROFILE=1` to enable it:
```sh
make bench PROFILE=1
mpirun -n 8 ./build/bin/bench
```

Profiling output is written as JSON under `out/`:
```
out/
├── bine_bcast_dhlv.json
├── bine_gather.json
├── bine_gatherv.json
├── bine_scatter.json
└── bine_scatterv.json
```

Omit `PROFILE=1` for a production build — all instrumentation compiles away with zero overhead.

---

## License

MIT License

Copyright (c) 2024 Daniele — Sapienza Università di Roma

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
