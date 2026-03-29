#include "bench/allgather.h"

#include "CollAlgo/allgather.h"
#include "CollBench/errors.h"
#include <mpi.h>
#include <stdlib.h>

#define BUFF_LEN 2

CB_Error_t CA_bench_bine_allgather(void) {
    CB_Error_t err = CB_SUCCESS;

    int *buff = NULL, *rbuff = NULL;
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), cleanup);
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size), cleanup);

    CB_MALLOC(buff, BUFF_LEN * sizeof(int), cleanup);
    CB_MALLOC(rbuff, BUFF_LEN * size * sizeof(int), cleanup);

    for(int i = 0; i < BUFF_LEN; i++)
        buff[i] = rank * BUFF_LEN + i;

    for(int i = 0; i < BUFF_LEN * size; i++)
        rbuff[i] = -1;

    MPI_CHECK(CA_bine_allgather_b2b(buff, BUFF_LEN, MPI_INT,
                                rbuff, BUFF_LEN, MPI_INT,
                                MPI_COMM_WORLD), cleanup);

    for(int i = 0; i < BUFF_LEN * size; i++) {
        if(rbuff[i] != i) {
            printf("[rank %d] MISMATCH at i=%d: got %d expected %d\n",
                   rank, i, rbuff[i], i);
            MPI_Abort(MPI_COMM_WORLD, 12);
        }
    }

    cleanup:
        free(rbuff);
        free(buff);
        return err;
}
