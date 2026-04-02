#include "CollAlgo/gather.h"
#include "CollAlgo/bine.h"
#include "CollAlgo/utils.h"
#include "CollBench/dist_list.h"
#include "CollBench/init.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int CA_bine_gatherv(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int *recvcounts, const int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm) {

    CB_COLL_START();

    assert(sendtype == recvtype);

    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &dtsize);

    if(!CA_is_pow_2(size)) {
        return MPI_ERR_ASSERT;
    }

    // Allocate full buffer data
    if (rank != root)
        CA_MALLOC(recvbuff, dtsize * (displs[size - 1] + recvcounts[size - 1]));

    // Copy the local data into the recvbuff
    memcpy((char*) recvbuff + (displs[rank] * dtsize), sendbuff, sendcount * dtsize);

    int s = log2(size);
    size_t min_block = rank, max_block = rank;
    int mod_rank = CA_mod(rank - root, size);
    int nb_rank = CA_rank2nb(mod_rank, s);

    int direction = (rank % 2) ? -1 : 1;

    int mask = 0x1;
    while (mask < size) {

        /**
         * INFO: Distance doubling pattern:
         * We keep xor-ing with increasing masks of ones
         * (rank ^ 0b1, rank ^ 0b11, rank ^ 0b111, ...)
         * so peer is the rank that differs by step least signicant bits
         */
        int mask_peer = (mask << 1) - 1;
        int nb_peer = nb_rank ^ mask_peer;
        int mod_peer = CA_nb2rank(nb_peer, s);
        int peer = CA_mod(mod_peer + root, size);

        /**
         * INFO: We use a << 2 mask as we want to skip the recv of leaves.
         */
        int mask_lsbs = (mask << 2) - 1;
        int lsbs = nb_rank & mask_lsbs;
        int eq_lsbs = (lsbs == 0 || lsbs == mask_lsbs);
        
        /**
         * INFO: If next iteration i will be root of my subtree then i send and exit else i recv,
         * as the communication is offsetted by 1 as we do not consider actual leaves as step = 0 roots
         */
        if (!eq_lsbs || ((mask << 1) >= size && (rank != root))) {
            if (max_block >= min_block) {
                /* Non-wrapped: contiguous range [min_block, max_block] */
                CB_LSEND(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff + (displs[min_block] * dtsize),
                    displs[max_block] - displs[min_block] + recvcounts[max_block],
                    sendtype,
                    peer, 0, comm
                );
            } else {
                /* Wrapped: upper chunk [min_block, size-1] then lower chunk [0, max_block] */
                CB_LSEND(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff + (displs[min_block] * dtsize),
                    displs[size - 1] - displs[min_block] + recvcounts[size - 1],
                    sendtype,
                    peer, 0, comm
                );

                CB_LSEND(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff,
                    displs[max_block] + recvcounts[max_block],
                    sendtype,
                    peer, 0, comm
                );
            }
            break;
        } else {
            size_t recv_start, recv_end;

            if (direction == 1) {
                recv_start = CA_mod(max_block + 1, size);
                recv_end   = CA_mod(max_block + mask, size);
                max_block  = recv_end;
            } else {
                recv_end   = CA_mod(min_block - 1, size);
                recv_start = CA_mod(min_block - mask, size);
                min_block  = recv_start;
            }

            if (recv_end >= recv_start) {
                /* Non-wrapped recv: contiguous range [recv_start, recv_end] */
                CB_LRECV(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff + (displs[recv_start] * dtsize),
                    displs[recv_end] - displs[recv_start] + recvcounts[recv_end],
                    recvtype,
                    peer, 0, comm
                );
            } else {
                /* Wrapped recv: upper chunk [recv_start, size-1] then lower chunk [0, recv_end] */
                CB_LRECV(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff + (displs[recv_start] * dtsize),
                    displs[size - 1] - displs[recv_start] + recvcounts[size - 1],
                    recvtype,
                    peer, 0, comm
                );

                CB_LRECV(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff,
                    displs[recv_end] + recvcounts[recv_end],
                    recvtype,
                    peer, 0, comm
                );
            }

            direction *= -1;
        }

        mask <<= 1;
    }

    if (rank != root)
        free(recvbuff);

    CB_COLL_END(comm, rank, root, "out/tree/bine_gatherv.json");

    return MPI_SUCCESS;
}

int CA_bine_gather(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {

    CB_COLL_START();

    assert(sendtype == recvtype);

    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &dtsize);

    if(!CA_is_pow_2(size)) {
        return MPI_ERR_ASSERT;
    }

    if (rank != root)
        CA_MALLOC(recvbuff, size * dtsize * recvcount);

    memcpy((char*) recvbuff + (rank * recvcount * dtsize), sendbuff, sendcount * dtsize);

    int s = log2(size);
    size_t min_block = rank, max_block = rank;
    int mod_rank = CA_mod(rank - root, size);
    int nb_rank = CA_rank2nb(mod_rank, s);

    int direction = (rank % 2) ? -1 : 1;

    int mask = 0x1;
    while (mask < size) {
        int mask_peer = (mask << 1) - 1;
        int nb_peer = nb_rank ^ mask_peer;
        int mod_peer = CA_nb2rank(nb_peer, s);
        int peer = CA_mod(mod_peer + root, size);

        int mask_lsbs = (mask << 2) - 1;
        int lsbs = nb_rank & mask_lsbs;
        int eq_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

        if (!eq_lsbs || ((mask << 1) >= size && (rank != root))) {
            if (max_block >= min_block) {
                /* Non-wrapped: contiguous range [min_block, max_block] */
                CB_LSEND(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff + (recvcount * min_block * dtsize),
                    recvcount * (max_block - min_block + 1),
                    sendtype,
                    peer, 0, comm
                );
            } else {
                /* Wrapped: upper chunk [min_block, size-1] then lower chunk [0, max_block] */
                CB_LSEND(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff + (recvcount * min_block * dtsize),
                    recvcount * (size - min_block),
                    sendtype,
                    peer, 0, comm
                );

                CB_LSEND(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff,
                    (max_block + 1) * recvcount,
                    sendtype,
                    peer, 0, comm
                );
            }
            break;
        } else {
            size_t recv_start, recv_end;

            if (direction == 1) {
                recv_start = CA_mod(max_block + 1, size);
                recv_end   = CA_mod(max_block + mask, size);
                max_block  = recv_end;
            } else {
                recv_end   = CA_mod(min_block - 1, size);
                recv_start = CA_mod(min_block - mask, size);
                min_block  = recv_start;
            }

            if (recv_end >= recv_start) {
                /* Non-wrapped recv: contiguous range [recv_start, recv_end] */
                CB_LRECV(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff + (recv_start * recvcount * dtsize),
                    (recv_end - recv_start + 1) * recvcount,
                    recvtype,
                    peer, 0, comm
                );
            } else {
                /* Wrapped recv: upper chunk [recv_start, size-1] then lower chunk [0, recv_end] */
                CB_LRECV(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff + (recvcount * recv_start * dtsize),
                    recvcount * (size - recv_start),
                    recvtype,
                    peer, 0, comm
                );

                CB_LRECV(
                    rank,
                    __builtin_popcount(mask_peer),
                    (char*) recvbuff,
                    recvcount * (recv_end + 1),
                    recvtype,
                    peer, 0, comm
                );
            }

            direction *= -1;
        }

        mask <<= 1;
    }

    if (rank != root)
        free(recvbuff);

    CB_COLL_END(comm, rank, root, "out/tree/bine_gather.json");

    return MPI_SUCCESS;
}
