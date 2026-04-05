#include "bench/reduce.h"
#include "CollAlgo/reduce.h"
#include "CollBench/errors.h"
#include <mpi.h>

#define BUFF_SIZE 100000
#define ROOT 0

int exp_at_idx(int idx, int size) {
    int exp = 0;
    for(int i = 0; i < size; i++) {
        exp += idx * (i + 1);
    }

    return exp;
}

CB_Error_t CA_bench_bine_reduce(void) {
    CB_Error_t err = CB_SUCCESS;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *send_buff = NULL;
    int *recv_buff = NULL;

    CB_MALLOC(send_buff, BUFF_SIZE * sizeof(int), cleanup);

    for(int i = 0; i < BUFF_SIZE; i++) {
        send_buff[i] = i * (rank + 1);
    }

    if(rank == ROOT) {
        CB_MALLOC(recv_buff, BUFF_SIZE * sizeof(int), cleanup);
    }

    CB_CHECK(CA_bine_reduce(send_buff, recv_buff, BUFF_SIZE, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD), cleanup);

    if(rank == ROOT) {
        for(int i = 0; i < BUFF_SIZE; i++) {
            if(recv_buff[i] != exp_at_idx(i, size)) {
                printf("[rank %d] MISMATCH at i=%d: got %d expected %d\n",
                    rank, i, recv_buff[i], i * rank);
                MPI_Abort(MPI_COMM_WORLD, 12);
            }
        }
    }

    cleanup:
        free(send_buff);
        if(rank == ROOT) {
            free(recv_buff);
        }

    return err;

}
