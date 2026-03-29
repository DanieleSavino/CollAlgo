#pragma once

#include <mpi.h>
#include <stdio.h>

#define CA_MALLOC(ptr, size) \
    do { \
        ptr = malloc(size); \
        if(!ptr) { \
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM); \
        } \
    } while(0) \

static inline int CA_mod(int a, int b){
    int r = a % b;
    return r < 0 ? r + b : r;
}

static inline int CA_is_pow_2(int num) {
    return num > 0 && (num & (num - 1)) == 0;
}

static inline void CA_print_rank_buff(const int *arr, const char *label, int len, int rank) {
    printf("[rank %d: %s] ", rank, label);
    for(int i = 0; i < len; i++)
        printf("%d ", arr[i]);
    printf("\n");
}
