#pragma once

#include <mpi.h>
#include "CollBench/errors.h"

#ifdef __cplusplus
extern "C" {
#endif

NODISCARD int CA_bine_allgather_b2b(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

#ifdef __cplusplus
}
#endif
