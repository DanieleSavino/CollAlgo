#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

int CA_cuda_reduce(const void *src, void *dst, int count, MPI_Datatype dt, MPI_Op op, int th_per_block);

#ifdef __cplusplus
}
#endif
