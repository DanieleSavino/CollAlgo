#pragma once

#include <mpi.h>

int CA_bine_gatherv(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm);

