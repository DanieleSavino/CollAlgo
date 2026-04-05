#pragma once

#include <mpi.h>
#include "CollBench/errors.h"

NODISCARD int CA_bine_alltoall(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

