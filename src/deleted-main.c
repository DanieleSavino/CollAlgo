#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "CollAlgo/gather.h"
#include "CollAlgo/bcast.h"
#include "CollBench/errors.h"
#include "CollBench/init.h"

#define ARR_SIZE 10
#define ROOT 2

void print_arr(int *arr, int size) {
    printf("[");
    for(int i = 0; i < size; i++) {
        printf("%d, ", arr[i]);
    }
    printf("]\n");
}

void gather_test(void) {
    MPI_Init(NULL, NULL);
    CB_init();

    CB_Error_t err = CB_SUCCESS;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *buff;
    CB_MALLOC(buff, sizeof(int) * ARR_SIZE, cleanup);

    for(int i = 0; i < ARR_SIZE; i++) {
        buff[i] = i;
    }

    int *rcounts = malloc(sizeof(int) * ARR_SIZE * size);
    int *displs = malloc(sizeof(int) * ARR_SIZE * size);
    int *rbuff = malloc(sizeof(int) * ARR_SIZE * size);

    for(int i = 0; i < size; i++) {
        rcounts[i] = ARR_SIZE;
        displs[i] = i * ARR_SIZE;
    }

    CA_bine_gatherv(buff, rcounts[rank], MPI_INT, rbuff, rcounts, displs, MPI_INT, ROOT, MPI_COMM_WORLD);

    if(rank == ROOT) {
        print_arr(rbuff, ARR_SIZE * size);
        for(int i = 0; i < ARR_SIZE; i++) {
            if(rbuff[i] != i % ARR_SIZE) {
                MPI_Abort(MPI_COMM_WORLD, 12);
            }
        }
    }

    printf("ok\n");

    cleanup:
        free(buff);
        CB_finalize();
        MPI_Finalize();
        return;
}

void bcast_test(void) {
    MPI_Init(NULL, NULL);
    CB_init();

    CB_Error_t err = CB_SUCCESS;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *buff;
    CB_MALLOC(buff, sizeof(int) * ARR_SIZE, cleanup);

    if(rank == ROOT) {
        for(int i = 0; i < ARR_SIZE; i++) {
            buff[i] = i;
        }

    }

    CA_bine_bcast_dhlv(buff, ARR_SIZE, MPI_INT, ROOT, MPI_COMM_WORLD);


    for(int i = 0; i < ARR_SIZE; i++) {
        if(buff[i] != i) {
            MPI_Abort(MPI_COMM_WORLD, 12);
        }
    }

    printf("ok\n");

    cleanup:
        free(buff);
        CB_finalize();
        MPI_Finalize();
        return;

}

int main(void) {
    bcast_test();
}
