#include "bench/gather.h"
#include "CollAlgo/gather.h"
#include "CollBench/errors.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define BUFF_LEN 12500 // 100,000 elems after gather
#define ROOT     0

CB_Error_t CA_bench_bine_gatherv(void) {
    CB_Error_t err = CB_SUCCESS;

    int *buff = NULL, *rbuff = NULL, *counts = NULL, *displs = NULL;

    int rank, size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), cleanup);
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size), cleanup);

    CB_MALLOC(counts, size * sizeof(int), cleanup);
    CB_MALLOC(displs, size * sizeof(int), cleanup);

    int offset = 0;
    for(int i = 0; i < size; i++) {
        counts[i] = BUFF_LEN + i;
        displs[i] = offset;
        offset   += counts[i];
    }

    int total = displs[size - 1] + counts[size - 1];
    CB_MALLOC(buff, counts[rank] * sizeof(int), cleanup);
    if(rank == ROOT)
        CB_MALLOC(rbuff, total * sizeof(int), cleanup);

    for(int i = 0; i < counts[rank]; i++)
        buff[i] = i;

    if(rank == ROOT)
        for(int i = 0; i < total; i++)
            rbuff[i] = -1;

    MPI_CHECK(CA_bine_gatherv(buff, counts[rank], MPI_INT,
                              rbuff, counts, displs, MPI_INT,
                              ROOT, MPI_COMM_WORLD), cleanup);

    if(rank == ROOT) {
        for(int r = 0; r < size; r++) {
            for(int i = 0; i < counts[r]; i++) {
                if(rbuff[displs[r] + i] != i) {
                    printf("[root] MISMATCH rank=%d i=%d: got %d expected %d\n",
                           r, i, rbuff[displs[r] + i], i);
                    MPI_Abort(MPI_COMM_WORLD, 12);
                }
            }
        }
    }

cleanup:
    free(rbuff);
    free(buff);
    free(counts);
    free(displs);
    return err;
}

CB_Error_t CA_bench_bine_gather(void) {
    CB_Error_t err = CB_SUCCESS;

    int *buff = NULL, *rbuff = NULL;

    int rank, size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), cleanup);
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size), cleanup);

    CB_MALLOC(buff, BUFF_LEN * sizeof(int), cleanup);
    if(rank == ROOT)
        CB_MALLOC(rbuff, BUFF_LEN * size * sizeof(int), cleanup);

    for(int i = 0; i < BUFF_LEN; i++)
        buff[i] = i;

    if(rank == ROOT)
        for(int i = 0; i < BUFF_LEN * size; i++)
            rbuff[i] = -1;

    MPI_CHECK(CA_bine_gather(buff, BUFF_LEN, MPI_INT,
                             rbuff, BUFF_LEN, MPI_INT,
                             ROOT, MPI_COMM_WORLD), cleanup);

    if(rank == ROOT) {
        for(int i = 0; i < BUFF_LEN * size; i++) {
            if(rbuff[i] != i % BUFF_LEN) {
                printf("[root] MISMATCH at i=%d: got %d expected %d\n",
                       i, rbuff[i], i % BUFF_LEN);
                MPI_Abort(MPI_COMM_WORLD, 12);
            }
        }
    }

cleanup:
    free(rbuff);
    free(buff);
    return err;
}
