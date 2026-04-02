#pragma once

#include <mpi.h>

int CA_bine_reduce(const void *sendbuff, void *recvbuff, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
