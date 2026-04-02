#include "bench/alltoall.h"
#include "CollAlgo/alltoall.h"
#include "CollBench/errors.h"
#include <stdlib.h>
#include <mpi.h>

#define SEND_LEN 12500 // 100,000 elems after allgather for 8 nodes

CB_Error_t CA_bench_bine_alltoall(void) {
    CB_Error_t err = CB_SUCCESS;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int BUFF_LEN = SEND_LEN * size;

    int *buff = NULL, *rbuff = NULL;
    CB_MALLOC(buff,  BUFF_LEN * sizeof(int), cleanup);
    CB_MALLOC(rbuff, BUFF_LEN * sizeof(int), cleanup);

    for(int i = 0; i < BUFF_LEN; i++) {
        buff[i] = i * rank;
    }

    MPI_CHECK(CA_bine_alltoall(buff, SEND_LEN, MPI_INT, rbuff, SEND_LEN, MPI_INT, MPI_COMM_WORLD), cleanup);

    for(int src = 0; src < size; src++) {
        for(int j = 0; j < SEND_LEN; j++) {
            int expected = (rank * SEND_LEN + j) * src;
            int got = rbuff[src * SEND_LEN + j];
            if(got != expected) {
                printf("[rank %d] MISMATCH at src=%d j=%d: got %d expected %d\n",
                       rank, src, j, got, expected);
                MPI_Abort(MPI_COMM_WORLD, 12);
            }
        }
    }

    cleanup:
        free(buff);
        free(rbuff);
        return err;
}
