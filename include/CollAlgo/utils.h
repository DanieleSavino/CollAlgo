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

/**
 * @brief Allocates memory and aborts the MPI job on failure.
 *
 * Wraps @c malloc with an MPI-aware error handler. If the allocation fails,
 * @c MPI_Abort is called on @c MPI_COMM_WORLD with @c MPI_ERR_NO_MEM,
 * terminating all ranks cleanly instead of continuing with a NULL pointer.
 *
 * @param ptr  L-value of pointer type that will receive the allocated address.
 * @param size Number of bytes to allocate.
 *
 * @note Expands to a @c do { } @c while(0) block so it is safe to use as a
 *       statement in all contexts (e.g., after a bare @c if).
 *
 * @warning @p ptr must be a valid lvalue — passing an expression with
 *          side-effects may produce unexpected behaviour.
 */
#define CA_MALLOC(ptr, size) \
    do { \
        ptr = malloc(size); \
        if(!ptr) { \
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM); \
        } \
    } while(0)

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

#define CA_MPI_CHECK(call, label) \
    int _mpi_err = call; \
    if(_mpi_err) \
        return _mpi_err
