#include "bench/bcast.h"

#include "CollAlgo/bcast.h"
#include "CollBench/errors.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define BUFF_LEN 100
#define ROOT 0

CB_Error_t CA_bench_bine_bcast_dhlv(void) {
    CB_Error_t err = CB_SUCCESS;

    int rank, size, *buff = NULL;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), cleanup);
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size), cleanup);

    CB_MALLOC(buff, BUFF_LEN * sizeof(int), cleanup);

    for(int i = 0; i < BUFF_LEN; i++) {
        buff[i] = (rank == ROOT) ? i : 0;
    }

    MPI_CHECK(CA_bine_bcast_dhlv(buff, BUFF_LEN, MPI_INT, ROOT, MPI_COMM_WORLD), cleanup);

    for(int i = 0; i < BUFF_LEN; i++) {
        if(buff[i] != i) {
            return CB_ERR_MPI;
        }
    }

    cleanup:
        free(buff);
        return err;
}
