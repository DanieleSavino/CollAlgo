#include "CollAlgo/allgather.h"
#include "CollAlgo/utils.h"
#include "CollAlgo/bine.h"
#include "CollBench/dist_list.h"
#include "CollBench/init.h"
#include <assert.h>
#include <mpi.h>
#include <stddef.h>
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

        // WARN: Prevent self communication by starting at 1
        for(int block = 1; block < size; block++) {
            int k = 31 - __builtin_clz(CA_nu(block, size));

            if(k == step) {
                int b2send, b2recv;

                /**
                 * INFO: Suppose we have a b2recv = rank
                 * then we would have rank + block = rank (mod. size),
                 * so we have that block = size * k | k in N:
                 *  - k = 0 is prevented by the loop idx starting at 1.
                 *  - k >= 1 is prevented by the last block being size - 1.
                 * Same can be said for b2send.
                 */
                if(rank % 2 == 0) {
                    b2recv = CA_mod(rank + block, size);
                    b2send = CA_mod(peer - block, size);
                }
                else {
                    b2recv = CA_mod(rank - block, size);
                    b2send = CA_mod(peer + block, size);
                }

                // Send to my peer a block from the local buffer offsetted by:
                // b2send * size of block,
                CB_ILSEND(
                    rank,
                    step,
                    (char*)recvbuff + (b2send * sendcount * dtsize),
                    sendcount,
                    sendtype,
                    peer,
                    0, comm, &reqs[req_idx++]
                );

                // receive from my peer a block and put it in the local buffer offset by:
                // b2send * size of block,
                CB_ILRECV(
                    rank,
                    step,
                    (char*)recvbuff + (b2recv * recvcount * dtsize),
                    recvcount,
                    recvtype,
                    peer,
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
