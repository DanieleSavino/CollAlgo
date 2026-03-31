#include "CollAlgo/allgather.h"
#include "CollAlgo/utils.h"
#include "CollAlgo/bine.h"
#include "CollBench/dist_list.h"
#include "CollBench/init.h"
#include <assert.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

int CA_bine_allgather_b2b(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {

    CB_COLL_START();

    assert(sendcount == recvcount);
    assert(recvtype == sendtype);


    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &dtsize);

    if(size % 2) {
        return MPI_ERR_ASSERT;
    }

    memcpy((char*) recvbuff + rank * recvcount * dtsize, sendbuff, sendcount * dtsize);

    int req_idx = 0;
    MPI_Request *reqs = NULL;
    CA_MALLOC(reqs, size * 2 * sizeof(MPI_Request));

    int s = CA_log2(size);
    int inv_mask = 1 << (s - 1);
    int step = 0;

    while (inv_mask > 0) {
        int peer;
        int mask_lsbs = (inv_mask << 1) - 1;

        if(rank % 2 == 0) {
            peer = CA_mod(rank + CA_nb2rank_raw(mask_lsbs), size);
        }
        else {
            peer = CA_mod(rank - CA_nb2rank_raw(mask_lsbs), size);
        }

        // 0 never sends 0
        for(int block = 1; block < size; block++) {
            int k = 31 - __builtin_clz(CA_nu(block, size));

            if(k == step || block == 0) {
                int b2send, b2recv;

                if(rank % 2 == 0) {
                    b2recv = CA_mod(rank + block, size);
                    b2send = CA_mod(peer - block, size);
                }
                else {
                    b2recv = CA_mod(rank - block, size);
                    b2send = CA_mod(peer + block, size);
                }

                // TODO: Why?
                int peer_send = (b2send != peer) ? peer : MPI_PROC_NULL;
                int peer_recv = (b2recv != rank) ? peer : MPI_PROC_NULL;

                CB_ILSEND(
                    rank,
                    step,
                    (char*)recvbuff + (b2send * sendcount * dtsize),
                    sendcount,
                    sendtype,
                    peer_send,
                    0, comm, &reqs[req_idx++]
                );

                CB_ILRECV(
                    rank,
                    step,
                    (char*)recvbuff + (b2recv * recvcount * dtsize),
                    recvcount,
                    recvtype,
                    peer_recv,
                    0, comm, &reqs[req_idx++]
                );
            }
        }
        inv_mask >>= 1;
        step++;


        CB_LWAITALL(reqs, req_idx);
        req_idx = 0;
    }

    free(reqs);

    CB_COLL_END(comm, rank, 0, "out/butterfly/bine_allgather_b2b.json");

    return MPI_SUCCESS;
}
