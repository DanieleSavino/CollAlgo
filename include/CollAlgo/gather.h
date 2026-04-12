#pragma once

#include <mpi.h>
#include "CollBench/errors.h"

#ifdef __cplusplus
extern "C" {
#endif

NODISCARD int CA_bine_gatherv(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm);
NODISCARD int CA_bine_gather(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

#ifdef __cplusplus
}
#endif
