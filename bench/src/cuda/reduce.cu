#include "bench/cuda/reduce.h"
#include "CollAlgo/reduce.h"
#include <mpi.h>

#define BUFF_SIZE 100000
#define ROOT 0

static __global__ void fill_kernel(int *buff, int rank, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buff[i] = i * (rank + 1);
}

static __global__ void check_kernel(int *buff, int size, int n, int *errors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && buff[i] != i * size * (size + 1) / 2)
        atomicAdd(errors, 1);
}

void CA_bench_reduce_fill_cuda(int *buff, int rank, int n) {
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;

        fill_kernel<<<blocks, threads>>>(buff, rank, n);
}

void CA_bench_reduce_check_cuda(int *buff, int size, int n, int *errors) {
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;

        check_kernel<<<blocks, threads>>>(buff, size, n, errors);
}

CB_Error_t CA_bench_bine_reduce_cuda(void) {
    CB_Error_t err = CB_SUCCESS;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *send_buff = NULL;
    int *recv_buff = NULL;
    int *d_errors  = NULL;

    cudaMalloc((void **)&send_buff, BUFF_SIZE * sizeof(int));
    CA_bench_reduce_fill_cuda(send_buff, rank, BUFF_SIZE);

    if (rank == ROOT)
        cudaMalloc((void **)&recv_buff, BUFF_SIZE * sizeof(int));

    int _ = CA_bine_reduce(send_buff, recv_buff, BUFF_SIZE, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        cudaMalloc((void **)&d_errors, sizeof(int));
        cudaMemset(d_errors, 0, sizeof(int));
        CA_bench_reduce_check_cuda(recv_buff, size, BUFF_SIZE, d_errors);
        int h_errors = 0;
        cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_errors) {
            printf("[rank %d] %d mismatches in bine_reduce\n", rank, h_errors);
            MPI_Abort(MPI_COMM_WORLD, 12);
        }
    }

cleanup:
    cudaFree(send_buff);
    if (rank == ROOT) {
        cudaFree(recv_buff);
        cudaFree(d_errors);
    }
    return err;
}
