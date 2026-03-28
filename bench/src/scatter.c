#include "bench/scatter.h"
#include "CollAlgo/scatter.h"
#include "CollBench/errors.h"
#include <mpi.h>
#include <stdlib.h>

#define BUFF_SIZE 40000000
#define ROOT 0

CB_Error_t CA_bench_bine_scatterv(void) {
    return CB_ERR_INVALID_ARG;
}

CB_Error_t CA_bench_bine_scatter(void) {
    CB_Error_t err = CB_SUCCESS;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *buff = NULL, *rbuff = NULL;
    CB_MALLOC(buff, BUFF_SIZE * sizeof(int), cleanup);
    CB_MALLOC(rbuff, (BUFF_SIZE / size) * sizeof(int), cleanup);

    for(int i = 0; i < BUFF_SIZE; i++) {
        buff[i] = i;
    }

    MPI_CHECK(CA_bine_scatter(buff, BUFF_SIZE / size, MPI_INT, rbuff, BUFF_SIZE / size, MPI_INT, ROOT, MPI_COMM_WORLD), cleanup);

    for(int i = 0; i < BUFF_SIZE / size; i++) {
        if(rbuff[i] != buff[ i + rank * (BUFF_SIZE / size) ]) {
            MPI_Abort(MPI_COMM_WORLD, 12);
        }
    }

    cleanup:
        free(buff);
        free(rbuff);
        return err;
}
