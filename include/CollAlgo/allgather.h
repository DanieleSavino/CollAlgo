#pragma once

#include <mpi.h>

int CA_bine_allgather_b2b(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
