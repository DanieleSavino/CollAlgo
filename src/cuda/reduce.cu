#include "CollAlgo/cuda/reduce.h"

struct OpSum  { template<typename T> __device__ static T apply(T a, T b) { return a + b; } };
struct OpProd { template<typename T> __device__ static T apply(T a, T b) { return a * b; } };
struct OpMax  { template<typename T> __device__ static T apply(T a, T b) { return a > b ? a : b; } };
struct OpMin  { template<typename T> __device__ static T apply(T a, T b) { return a < b ? a : b; } };
struct OpBand { template<typename T> __device__ static T apply(T a, T b) { 
    if constexpr (std::is_integral_v<T>) return a & b;
    else return a;
} };
struct OpBor  { template<typename T> __device__ static T apply(T a, T b) { 
    if constexpr (std::is_integral_v<T>) return a | b;
    else return a;
} };
struct OpBxor { template<typename T> __device__ static T apply(T a, T b) { 
    if constexpr (std::is_integral_v<T>) return a ^ b;
    else return a;
} };

template<typename T, typename Op>
__global__ void reduce_kernel(const T *src, T *dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = Op::apply(dst[i], src[i]);
}

#define LAUNCH(T, Op) \
    reduce_kernel<T, Op><<<grid, th_per_block, 0>>>((const T*)src, (T*)dst, count)

#define DISPATCH_TYPE(Op)                              \
    if      (dt == MPI_INT)    { LAUNCH(int,    Op); } \
    else if (dt == MPI_FLOAT)  { LAUNCH(float,  Op); } \
    else if (dt == MPI_DOUBLE) { LAUNCH(double, Op); } \
    else if (dt == MPI_LONG)   { LAUNCH(long,   Op); } \
    else return 1;               \
    return 0

int CA_cuda_reduce(const void *src, void *dst, int count,
                           MPI_Datatype dt, MPI_Op op, int th_per_block) {
    int grid = (count + th_per_block - 1) / th_per_block;

    if      (op == MPI_SUM)  { DISPATCH_TYPE(OpSum);  }
    else if (op == MPI_PROD) { DISPATCH_TYPE(OpProd); }
    else if (op == MPI_MAX)  { DISPATCH_TYPE(OpMax);  }
    else if (op == MPI_MIN)  { DISPATCH_TYPE(OpMin);  }
    else if (op == MPI_BAND) { DISPATCH_TYPE(OpBand); }
    else if (op == MPI_BOR)  { DISPATCH_TYPE(OpBor);  }
    else if (op == MPI_BXOR) { DISPATCH_TYPE(OpBxor); }
    else return 1;

    cudaDeviceSynchronize();
}
