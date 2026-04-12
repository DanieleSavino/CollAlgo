#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int _CA_cuda_is_devptr(const void *ptr);

#ifdef __cplusplus
}
#endif
