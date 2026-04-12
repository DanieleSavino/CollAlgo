#pragma once
#include "CollBench/errors.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

CB_Error_t CA_bench_bine_reduce_cuda(void);

#ifdef __cplusplus
}
#endif
