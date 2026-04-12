#pragma once

#include <mpi.h>
#include "CollBench/errors.h"

#ifdef __cplusplus
extern "C" {
#endif

NODISCARD int CA_bine_bcast_dhlv(void *buff, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

#ifdef __cplusplus
}
#endif
