/**
 * @file utils.h
 * @brief Utility macros and inline helpers for MPI-based cellular automata.
 *
 * Provides memory allocation with MPI-aware error handling, modular arithmetic
 * with non-negative guarantees, power-of-two detection, and rank-aware debug
 * printing utilities intended for toroidal grid simulations.
 *
 * @author DanieleSavino <savino.2140356@studenti.uniroma1.it>
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Daniele Savino - Sapienza Università di Roma
 */

#pragma once
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the non-negative remainder of @p a divided by @p b.
 *
 * Unlike the built-in @c % operator, this function always returns a value in
 * the range @c [0, b), even when @p a is negative. This is useful for wrapping
 * grid indices in toroidal (periodic-boundary) cellular automata.
 *
 * @param a Dividend (may be negative).
 * @param b Divisor (must be positive).
 * @return  The mathematical modulus @c a mod @c b, guaranteed non-negative.
 *
 * @note Undefined behaviour if @p b is zero.
 */
static inline int CA_mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

/**
 * @brief Tests whether an integer is an exact power of two.
 *
 * Uses the classic bitmask trick: a positive power of two has exactly one bit
 * set, so @c (num & (num-1)) is zero if and only if @p num is a power of two.
 *
 * @param num Value to test.
 * @return    Non-zero (true) if @p num is a power of two; zero (false) otherwise.
 *            Returns zero for non-positive inputs.
 */
static inline int CA_is_pow_2(int num) {
    return num > 0 && (num & (num - 1)) == 0;
}

/**
 * @brief Prints a labelled integer array to @c stdout, prefixed by the MPI rank.
 *
 * Intended for per-rank debug output. Each element is printed space-separated
 * on a single line with the format:
 * @verbatim [rank <rank>: <label>] e0 e1 e2 ... \n @endverbatim
 *
 * @param arr   Pointer to the array of integers to print.
 * @param label Null-terminated descriptive label shown in the prefix.
 * @param len   Number of elements in @p arr.
 * @param rank  MPI rank of the calling process (used purely for display).
 *
 * @note All ranks that call this function will produce output; use
 *       @c CA_root_print if output from a single rank is sufficient.
 */
static inline void CA_print_rank_buff(const int *arr, const char *label, int len, int rank) {
    printf("[rank %d: %s] ", rank, label);
    for (int i = 0; i < len; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

/**
 * @brief Prints a message to @c stdout only on the designated root rank.
 *
 * A lightweight guard that suppresses output on all ranks except @p root,
 * avoiding interleaved output from multiple processes.
 *
 * @param msg  Null-terminated string to print (a newline is appended automatically).
 * @param rank MPI rank of the calling process.
 * @param root MPI rank that is permitted to print.
 */
static inline void CA_root_print(const char *msg, int rank, int root) {
    if (rank == root) {
        printf("%s\n", msg);
    }
}

#ifdef CA_CUDA
    #include "CollAlgo/cuda/utils.h"
#endif

static inline int CA_is_devptr(const void *ptr) {
#ifdef CA_CUDA
    return _CA_cuda_is_devptr(ptr);
#endif

    (void)ptr;
    return 0;
}


// ── Error-check macros ────────────────────────────────────────────────────────

#define CA_MPI_CHECK(call) \
    do { \
        int _err = (call); \
        if (_err != MPI_SUCCESS) { \
            fprintf(stderr, "[MPI error] %s:%d: %d\n", __FILE__, __LINE__, _err); \
            return _err; \
        } \
    } while(0)

#define CA_MPI_CHECK_LABEL(call, label) \
    do { \
        int _err = (call); \
        if (_err != MPI_SUCCESS) { \
            fprintf(stderr, "[MPI error] %s:%d: %d\n", __FILE__, __LINE__, _err); \
            goto label; \
        } \
    } while(0)

#ifdef CA_CUDA
#define CA_CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "[CUDA error] %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

#define CA_CUDA_CHECK_LABEL(call, label) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "[CUDA error] %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
            goto label; \
        } \
    } while(0)
#endif

// ── Allocation macros ─────────────────────────────────────────────────────────

#define CA_MALLOC(ptr, size) \
    do { \
        (ptr) = malloc(size); \
        if (!(ptr)) { \
            fprintf(stderr, "[OOM] %s:%d: malloc(%zu) failed\n", \
                __FILE__, __LINE__, (size_t)(size)); \
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM); \
        } \
    } while(0)

#define CA_MALLOC_LABEL(ptr, size, label) \
    do { \
        (ptr) = malloc(size); \
        if (!(ptr)) { \
            fprintf(stderr, "[OOM] %s:%d: malloc(%zu) failed\n", \
                __FILE__, __LINE__, (size_t)(size)); \
            goto label; \
        } \
    } while(0)

#ifdef CA_CUDA
#define CA_CUDA_MALLOC(ptr, size) \
    CA_CUDA_CHECK(cudaMalloc((void **)&(ptr), (size)))

#define CA_CUDA_MALLOC_LABEL(ptr, size, label) \
    CA_CUDA_CHECK_LABEL(cudaMalloc((void **)&(ptr), (size)), label)

// ── Unified host/device macros (require int use_cuda in scope) ────────────────

#define CA_UMALLOC(ptr, size, use_cuda) \
    do { \
        if (use_cuda) { CA_CUDA_MALLOC(ptr, size); } \
        else          { CA_MALLOC(ptr, size); } \
    } while(0)

#define CA_UMALLOC_LABEL(ptr, size, use_cuda, label) \
    do { \
        if (use_cuda) { CA_CUDA_MALLOC_LABEL(ptr, size, label); } \
        else          { CA_MALLOC_LABEL(ptr, size, label); } \
    } while(0)

#define CA_UFREE(ptr, use_cuda) \
    do { \
        if (use_cuda) { cudaFree(ptr); } \
        else          { free(ptr); } \
    } while(0)

#define CA_UMEMCPY(dst, src, size, use_cuda) \
    do { \
        if (use_cuda) { \
            CA_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)); \
        } else { \
            memcpy(dst, src, size); \
        } \
    } while(0)

#define CA_UMEMCPY_LABEL(dst, src, size, use_cuda, label) \
    do { \
        if (use_cuda) { \
            CA_CUDA_CHECK_LABEL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice), label); \
        } else { \
            memcpy(dst, src, size); \
        } \
    } while(0)

#else
// No-CUDA fallbacks — use_cuda is always 0 so these collapse to host ops
#define CA_UMALLOC(ptr, size, use_cuda)              CA_MALLOC(ptr, size)
#define CA_UMALLOC_LABEL(ptr, size, use_cuda, label) CA_MALLOC_LABEL(ptr, size, label)
#define CA_UFREE(ptr, use_cuda)                      free(ptr)
#define CA_UMEMCPY(dst, src, size, use_cuda)         memcpy(dst, src, size)
#define CA_UMEMCPY_LABEL(dst, src, size, use_cuda, label) memcpy(dst, src, size)
#endif

#ifdef __cplusplus
}
#endif
