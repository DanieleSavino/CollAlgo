#include "bench/gather.h"
#include "CollAlgo/gather.h"
#include "CollBench/errors.h"
#include <stdlib.h>
#include <mpi.h>

#define BUFF_LEN 100
#define ROOT 0

CB_Error_t CA_bench_bine_gatherv(void) {
    CB_Error_t err = CB_SUCCESS;

    int rank, size, *buff = NULL, *rbuff = NULL, *counts = NULL, *displs = NULL;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), cleanup);
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size), cleanup);

    CB_MALLOC(counts, size * sizeof(int), cleanup);
    CB_MALLOC(displs, size * sizeof(int), cleanup);

    int offset = 0;
    for(int i = 0; i < size; i++) {
        counts[i] = BUFF_LEN + i;
        displs[i] = offset;
        offset += BUFF_LEN + i;
    }

    CB_MALLOC(buff, counts[rank] * sizeof(int), cleanup);

    if(rank == ROOT) {
        CB_MALLOC(rbuff, (displs[size - 1] + counts[size - 1]) * sizeof(int), cleanup);
    }

    for(int i = 0; i < BUFF_LEN; i++) {
        buff[i] = i;
    }

    MPI_CHECK(CA_bine_gatherv(buff, counts[rank], MPI_INT, rbuff, counts, displs, MPI_INT, ROOT, MPI_COMM_WORLD), cleanup);

    for(int i = 0; i < BUFF_LEN; i++) {
        if(buff[i] != i % BUFF_LEN) {
            return CB_ERR_MPI;
        }
    }

    cleanup:
        free(rbuff);
        free(buff);
        free(counts);
        free(displs);
        return err;
}
