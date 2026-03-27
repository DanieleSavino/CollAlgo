#include <mpi.h>
#include <stdlib.h>
#include "CollAlgo/bcast.h"
#include "CollBench/errors.h"
#include "CollBench/init.h"

#define ARR_SIZE 100
#define ROOT 2

int main(void) {
    MPI_Init(NULL, NULL);
    CB_init();

    CB_Error_t err = CB_SUCCESS;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *buff;
    CB_MALLOC(buff, sizeof(int) * ARR_SIZE, cleanup);

    for(int i = 0; i < ARR_SIZE; i++) {
        buff[i] = (rank == ROOT) ? i : 0;
    }

    CA_bine_bcast_dhlv(buff, ARR_SIZE, MPI_INT, ROOT, MPI_COMM_WORLD);

    for(int i = 0; i < ARR_SIZE; i++) {
        if(buff[i] != i) {
            MPI_Abort(MPI_COMM_WORLD, 12);
        }
    }

    cleanup:
        free(buff);
        CB_finalize();
        MPI_Finalize();
        return err;
}
