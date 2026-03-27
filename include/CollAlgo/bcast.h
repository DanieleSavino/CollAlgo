#pragma once

#include <mpi.h>

int CA_bine_bcast_dhlv(void *buff, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
