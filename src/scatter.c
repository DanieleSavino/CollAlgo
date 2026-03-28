#include "CollAlgo/scatter.h"
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

int CA_bine_scatterv(const void *sendbuff, int *sendcounts, const int *displs, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
    CB_COLL_START();

    assert(sendtype == recvtype);

    (void)recvcount;

    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &dtsize);

    if(!CA_is_pow_2(size)) {
        return MPI_ERR_ASSERT;
    }

    int s = ceil(log2(size));

    char *tmpbuff = NULL, *sbuff = NULL, *rbuff = NULL;
    if(rank == root) {
        sbuff = (char*) sendbuff;
    }

    int mod_rank = CA_mod(rank - root, size);
    int nb_rank  = CA_rank2nb(mod_rank, s);
    int direction = (rank % 2) ? -1 : 1;
    if(s % 2 == 0) direction *= -1;

    int max_block, min_block;
    if(rank % 2 == 0) {
        max_block = CA_mod((rank + 0x55555555) & ((0x1 << s) - 1), size);
        min_block = CA_mod((rank - 0xAAAAAAAA) & ((0x1 << s) - 1), size);
    } else {
        min_block = CA_mod((rank - 0x55555555) & ((0x1 << s) - 1), size);
        max_block = CA_mod((rank + 0xAAAAAAAA) & ((0x1 << s) - 1), size);
    }

    int recvd           = (root == rank);
    int is_leaf         = 0;
    int last_recv_start = 0;   // global block index where tmpbuff starts

    int mask = 1 << (s - 1);

    while(mask > 0) {
        size_t send_start, send_end, recv_start, recv_end;

        int mask_lsbs = (mask << 1) - 1;
        int nb_peer   = nb_rank ^ mask_lsbs;
        int mod_peer  = CA_nb2rank(nb_peer, s);
        int peer      = CA_mod(mod_peer + root, size);

        int lsbs    = nb_rank & mask_lsbs;
        int eq_lsbs = (lsbs == 0 || lsbs == mask_lsbs);

        size_t top_start    = min_block;
        size_t top_end      = CA_mod(min_block + mask - 1, size);
        size_t bottom_start = CA_mod(top_end + 1, size);
        size_t bottom_end   = max_block;

        if(direction == 1) {
            send_start = bottom_start; send_end = bottom_end;
            recv_start = top_start;    recv_end = top_end;
            max_block  = CA_mod(max_block - mask, size);
        } else {
            send_start = top_start;    send_end = top_end;
            recv_start = bottom_start; recv_end = bottom_end;
            min_block  = CA_mod(min_block + mask, size);
        }

        if(recvd) {
            // send_start/send_end are local indices into sbuff.
            // Convert to global for displs lookup.
            int g_ss = last_recv_start + (int)send_start;
            int g_se = last_recv_start + (int)send_end;

            if(send_end >= send_start) {
                int sc      = displs[g_se] - displs[g_ss] + sendcounts[g_se];
                int buf_off = displs[g_ss] - displs[last_recv_start];
                CB_LSEND(rank, s - __builtin_popcount(mask_lsbs),
                    sbuff + buf_off * dtsize,
                    sc, sendtype, peer, 0, comm);
            } else {
                // wrapped: [send_start .. local_last] then [0 .. send_end]
                int local_last = (int)(max_block + (size_t)mask); // before update
                int g_last     = last_recv_start + local_last;
                int sc1        = displs[g_last] - displs[g_ss] + sendcounts[g_last];
                int sc2        = displs[g_se]   - displs[last_recv_start] + sendcounts[g_se];
                int buf_off    = displs[g_ss]   - displs[last_recv_start];
                CB_LSEND(rank, s - __builtin_popcount(mask_lsbs),
                    sbuff + buf_off * dtsize,
                    sc1, sendtype, peer, 0, comm);
                CB_LSEND(rank, s - __builtin_popcount(mask_lsbs),
                    sbuff,
                    sc2, sendtype, peer, 0, comm);
            }
        } else if(eq_lsbs) {
            size_t num_blocks = CA_mod(recv_end - recv_start + 1, size);

            if(recv_start == recv_end) {
                rbuff   = (char*) recvbuff;
                is_leaf = 1;
            } else {
                // Total bytes to receive = sum of sendcounts[recv_start..recv_end]
                int total = displs[recv_end] - displs[recv_start] + sendcounts[recv_end];
                CA_MALLOC(tmpbuff, total * dtsize);
                sbuff = tmpbuff;
                rbuff = tmpbuff;

                last_recv_start = (int)recv_start;
                min_block       = 0;
                max_block       = (int)num_blocks - 1;
            }

            if(recv_end >= recv_start) {
                // recv_start/recv_end are still global here (before remapping)
                int rc = displs[recv_end] - displs[recv_start] + sendcounts[recv_end];
                CB_LRECV(rank, s - __builtin_popcount(mask_lsbs),
                    rbuff, rc, recvtype, peer, 0, comm);
            } else {
                int rc1 = displs[size - 1] - displs[recv_start] + sendcounts[size - 1];
                int rc2 = displs[recv_end] + sendcounts[recv_end];
                CB_LRECV(rank, s - __builtin_popcount(mask_lsbs),
                    rbuff, rc1, recvtype, peer, 0, comm);
                CB_LRECV(rank, s - __builtin_popcount(mask_lsbs),
                    rbuff + rc1 * dtsize, rc2, recvtype, peer, 0, comm);
            }

            recvd = 1;
        }

        mask >>= 1;
        direction *= -1;
    }

    if(!is_leaf) {
        int local_offset = displs[rank] - displs[last_recv_start];
        memcpy((char*) recvbuff,
               (char*) sbuff + local_offset * dtsize,
               sendcounts[rank] * dtsize);
    }

    if(tmpbuff != NULL) free(tmpbuff);

    CB_COLL_END(comm, root, "out/bine_scatterv.json");
    return MPI_SUCCESS;
}

int CA_bine_scatter(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {

    CB_COLL_START();

    assert(sendcount == recvcount);
    assert(sendtype == recvtype);

    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &dtsize);

    if(!CA_is_pow_2(size)) {
        return MPI_ERR_ASSERT;
    }

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
                CA_MALLOC(tmpbuff, recvcount * num_blocks * dtsize);

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

    free(tmpbuff);

    CB_COLL_END(comm, root, "out/bine_scatter.json");

    return MPI_SUCCESS;
}
