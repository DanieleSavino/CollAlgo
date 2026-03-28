#include "bench/scatter.h"
#include "CollAlgo/scatter.h"
#include "CollBench/errors.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define BUFF_SIZE 4
#define ROOT      0

CB_Error_t CA_bench_bine_scatterv(void) {
    CB_Error_t err = CB_SUCCESS;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *buff = NULL, *rbuff = NULL, *counts = NULL, *displs = NULL;
    CB_MALLOC(counts, size * sizeof(int), cleanup);
    CB_MALLOC(displs, size * sizeof(int), cleanup);

    int offset = 0;
    for(int i = 0; i < size; i++) {
        counts[i] = BUFF_SIZE + i;
        displs[i] = offset;
        offset   += counts[i];
    }

    int total = displs[size - 1] + counts[size - 1];
    CB_MALLOC(buff,  total          * sizeof(int), cleanup);
    CB_MALLOC(rbuff, counts[rank]   * sizeof(int), cleanup);

    for(int r = 0; r < size; r++)
        for(int i = 0; i < counts[r]; i++)
            buff[displs[r] + i] = i * r;

    for(int i = 0; i < counts[rank]; i++)
        rbuff[i] = -1;

    MPI_CHECK(CA_bine_scatterv(buff, counts, displs, MPI_INT,
                               rbuff, counts[rank], MPI_INT,
                               ROOT, MPI_COMM_WORLD), cleanup);

    for(int i = 0; i < counts[rank]; i++) {
        if(rbuff[i] != i * rank) {
            printf("[rank %d] MISMATCH at i=%d: got %d expected %d\n",
                   rank, i, rbuff[i], i * rank);
            MPI_Abort(MPI_COMM_WORLD, 12);
        }
    }

cleanup:
    free(buff);
    free(rbuff);
    free(counts);
    free(displs);
    return err;
}

CB_Error_t CA_bench_bine_scatter(void) {
    CB_Error_t err = CB_SUCCESS;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = BUFF_SIZE / size;
    int *buff = NULL, *rbuff = NULL;
    CB_MALLOC(buff,  BUFF_SIZE * sizeof(int), cleanup);
    CB_MALLOC(rbuff, chunk     * sizeof(int), cleanup);

    for(int i = 0; i < BUFF_SIZE; i++)
        buff[i] = i;

    MPI_CHECK(CA_bine_scatter(buff, chunk, MPI_INT,
                              rbuff, chunk, MPI_INT,
                              ROOT, MPI_COMM_WORLD), cleanup);

    for(int i = 0; i < chunk; i++) {
        if(rbuff[i] != buff[i + rank * chunk]) {
            printf("[rank %d] MISMATCH at i=%d: got %d expected %d\n",
                   rank, i, rbuff[i], buff[i + rank * chunk]);
            MPI_Abort(MPI_COMM_WORLD, 12);
        }
    }

cleanup:
    free(buff);
    free(rbuff);
    return err;
}
