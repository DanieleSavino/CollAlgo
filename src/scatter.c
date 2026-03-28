#include "CollAlgo/scatter.h"
#include "CollAlgo/bine.h"
#include "CollBench/bench.h"
#include "CollBench/dist_list.h"
#include "CollBench/errors.h"
#include "CollBench/init.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

int CA_bine_scatterv(const void *sendbuff, int *sendcounts, const int *displs, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
    return MPI_UNDEFINED;
}

int CA_bine_scatter(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {

    CB_COLL_START();
    //TODO: add 2^n check

    assert(sendcount == recvcount);
    assert(sendtype == recvtype);

    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &dtsize);

    int s = ceil(log2(size));

    char *tmpbuff = NULL, *sbuff = NULL, *rbuff = NULL;
    if(rank == root) {
        sbuff = (char*) sendbuff;
    }

    int mod_rank = CA_mod(rank - root, size);
    int nb_rank = CA_rank2nb(mod_rank, s);
    int direction = (rank % 2) ? -1 : 1;

    if(s % 2 == 0) {
        direction *= -1;
    }


    // I need to do the opposite of what I did in the gather.
    // Thus, I need to know where min_resident_block and max_resident_block
    // ended up after the last step.
    // Even ranks added 2^0, 2^2, 2^4, ... to max_resident_block
    //   and subtracted 2^1, 2^3, 2^5, ... from min_resident_block
    // Odd ranks subtracted 2^0, 2^2, 2^4, ... from min_resident_block
    //      and added 2^1, 2^3, 2^5, ... to max_resident_block
    int max_block, min_block;
    if(rank % 2 == 0){
        max_block = CA_mod((rank + 0x55555555) & ((0x1 << (int) s) - 1), size);
        min_block = CA_mod((rank - 0xAAAAAAAA) & ((0x1 << (int) s) - 1), size);
    }
    else {
        min_block = CA_mod((rank - 0x55555555) & ((0x1 << (int) s) - 1), size);
        max_block = CA_mod((rank + 0xAAAAAAAA) & ((0x1 << (int) s) - 1), size);    
    }

    int send_off = rank;
    int recvd = root == rank;
    int is_leaf = 0;

    int mask = 1 << (s - 1);

    while(mask > 0) {
        size_t send_start, send_end, recv_start, recv_end;

        int mask_lsbs = (mask << 1) - 1;
        int nb_peer = nb_rank ^ mask_lsbs;
        int mod_peer = CA_nb2rank(nb_peer, s);
        int peer = CA_mod(mod_peer + root, size);

        int lsbs = nb_rank & mask_lsbs;
        int eq_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

        size_t top_start = min_block;
        size_t top_end = CA_mod(min_block + mask - 1, size);
        size_t bottom_start = CA_mod(top_end + 1, size);
        size_t bottom_end = max_block;

        if(direction == 1) {
            // Send bottom
            send_start = bottom_start;
            send_end = bottom_end;
            recv_start = top_start;
            recv_end = top_end;
            max_block = CA_mod(max_block - mask, size);
        }
        else {
            // Send top
            send_start = top_start;
            send_end = top_end;
            recv_start = bottom_start;
            recv_end = bottom_end;
            min_block = CA_mod(min_block + mask, size);
        }

        if(recvd) {
            if(send_end >= send_start) {
                CB_LSEND(
                    rank,
                    s - __builtin_popcount(mask_lsbs),
                    sbuff + (send_start * sendcount * dtsize),
                    sendcount * (send_end - send_start + 1),
                    sendtype,
                    peer,
                    0, comm
                );
            }
            else {
                CB_LSEND(
                    rank,
                    s - __builtin_popcount(mask_lsbs),
                    sbuff + (send_start * sendcount * dtsize),
                    sendcount * (size - send_start),
                    sendtype,
                    peer,
                    0, comm
                );

                CB_LSEND(
                    rank,
                    s - __builtin_popcount(mask_lsbs),
                    sbuff,
                    sendcount * (send_end + 1),
                    sendtype,
                    peer,
                    0, comm
                );

            }
        }
        else if(eq_lsbs) {

            size_t num_blocks = CA_mod(recv_end - recv_start + 1, size);
            if(recv_start == recv_end) {
                rbuff = (char*) recvbuff;
                is_leaf = 1;
            }
            else {
                tmpbuff = malloc(recvcount * num_blocks * dtsize);

                sbuff = tmpbuff; rbuff = tmpbuff;

                min_block = 0;
                max_block = num_blocks - 1;

                send_off = CA_mod(rank - recv_start, size);
            }

            if(recv_end >= recv_start) {
                CB_LRECV(
                    rank,
                    s - __builtin_popcount(mask_lsbs),
                    rbuff,
                    recvcount * num_blocks,
                    recvtype,
                    peer,
                    0, comm
                );
            }
            else {
                CB_LRECV(
                    rank,
                    s - __builtin_popcount(mask_lsbs),
                    rbuff,
                    recvcount * (size - recv_start),
                    recvtype,
                    peer,
                    0, comm
                );

                CB_LRECV(
                    rank,
                    s - __builtin_popcount(mask_lsbs),
                    rbuff + (recvcount * (size - recv_start) * dtsize),
                    recvcount * (recv_end + 1),
                    recvtype,
                    peer,
                    0, comm
                );
            }

            recvd = 1;
        }

        mask >>= 1;
        direction *= -1;
    }

    if(!is_leaf) {
        memcpy((char*) recvbuff, (char*) sbuff + (send_off * recvcount * dtsize), recvcount * dtsize);
    }

    CB_COLL_END(comm, root, "out/bine_scatter.json");

    free(tmpbuff);
    return MPI_SUCCESS;
}
