#pragma once

#include <mpi.h>
#include "CollBench/errors.h"

#ifdef __cplusplus
extern "C" {
#endif

NODISCARD int CA_bine_reduce(const void *sendbuff, void *recvbuff, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

#ifdef __cplusplus
}
#endif
