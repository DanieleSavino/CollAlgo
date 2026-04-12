#include <cuda_runtime.h>
#include "bench/cuda/bcast.h"
#include "CollAlgo/bcast.h"
#include "CollAlgo/utils.h"
#include <mpi.h>

#define BUFF_LEN 100000
#define ROOT 0

static __global__ void fill_kernel(int *buff, int n, int is_root) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n) buff[i] = (is_root ? i : 0);
}

static __global__ void check_kernel(int *buff, int n, int *errors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n && buff[i] != i) {
        atomicAdd(errors, 1);
    }
}

NODISCARD CB_Error_t CA_bench_bine_bcast_dhlv_cuda(void) {
    CB_Error_t err = CB_SUCCESS;

    int rank, size, *buff = NULL;
    CA_MPI_CHECK_ABORT(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CA_MPI_CHECK_ABORT(MPI_Comm_size(MPI_COMM_WORLD, &size));

    CA_CUDA_MALLOC(buff, BUFF_LEN * sizeof(int));

    int threads = 256;
    int blocks  = (BUFF_LEN + threads - 1) / threads;

    fill_kernel<<<blocks, threads>>>(buff, BUFF_LEN, rank == ROOT);

    MPI_CHECK(CA_bine_bcast_dhlv(buff, BUFF_LEN, MPI_INT, ROOT, MPI_COMM_WORLD), cleanup);

    int *d_errors, h_errors;
    CA_CUDA_MALLOC(d_errors, sizeof(int));
    check_kernel<<<blocks, threads>>>(buff, BUFF_LEN, d_errors);

    CA_CUDA_CHECK(cudaDeviceSynchronize());

    CA_CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));

    if(h_errors > 0) {
        return CB_ERR_MPI;
    }

    cleanup:
        free(buff);
        return err;
}
