#include "CollAlgo/cuda/utils.h"

int _CA_cuda_is_devptr(const void *ptr) {
    struct cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) return 0;
    return attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged;
}
