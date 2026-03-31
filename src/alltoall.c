#include "CollAlgo/alltoall.h"
#include "CollAlgo/utils.h"
#include "CollAlgo/bine.h"
#include "CollBench/dist_list.h"
#include "CollBench/init.h"
#include <assert.h>
#include <mpi.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>

int CA_bine_alltoall(const void *sendbuff, int sendcount, MPI_Datatype sendtype, void *recvbuff, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {

    CB_COLL_START();

    assert(sendcount == recvcount);
    assert(sendtype == recvtype);

    //TODO: check non pow of 2

    int rank, size, dtsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Type_size(sendtype, &dtsize);

    int num_blocks = size;
    int num_blocks_next = 0;
    int sbuff_size = sendcount * dtsize;
    int tmpbuff_size = sbuff_size * size;
    int tmpbuff_size_real = tmpbuff_size + (2 * sizeof(unsigned int) * size);

    char *tmpbuff;
    CA_MALLOC(tmpbuff, tmpbuff_size_real);

    unsigned int *block = (unsigned int *)(tmpbuff + tmpbuff_size);
    unsigned int *block_next = (unsigned int *)(tmpbuff + tmpbuff_size + sizeof(unsigned int) * size);

    for(int i = 0; i < size; i++) {
        block[i] = i;
    }

    memcpy(tmpbuff, sendbuff, tmpbuff_size);

    int min_block_s, max_block_s;

    int mask = 0x1;
    int inv_mask = 0x1 << (CA_log2(size) - 1);
    int block_first_mask = ~(inv_mask - 1);

    while (mask < size) {
        int lsbs = CA_nb2rank_raw((mask << 1) - 1);

        int peer;
        if(rank % 2 == 0) {
            peer = CA_mod(rank + lsbs, size);
        }
        else {
            peer = CA_mod(rank - lsbs, size);
        }

        min_block_s = CA_remap_rank(size, peer) & block_first_mask;
        max_block_s = min_block_s + inv_mask - 1;

        int recv_blocks = 0, send_blocks = 0;
        int off_send = 0, off_keep = 0;
        num_blocks_next = 0;
        for(int i = 0; i < size; i++) {
            unsigned int b = block[i % num_blocks];

            unsigned int remap_b = CA_remap_rank(size, b);
            int off = i * sbuff_size;

            // Move the blocks to keep at the beginning of tmpbuff
            // and the ones i want to send to recvbuff
            if(remap_b >= (unsigned int)min_block_s && remap_b <= (unsigned int)max_block_s) {
                memcpy((char*) recvbuff + off_send, tmpbuff + off, sbuff_size);
                off_send += sbuff_size;
                send_blocks++;
            }
            else {
                if(off != off_keep) {
                    memcpy(tmpbuff + off_keep, tmpbuff + off, sbuff_size);
                }
                off_keep += sbuff_size;
                recv_blocks++;

                block_next[num_blocks_next] = b;
                num_blocks_next++;
            }
        }

        assert(recv_blocks == size/2);
        assert(send_blocks == size/2);
        num_blocks /= 2;

        MPI_Request reqs[2];
        CB_ILSEND(rank, CA_log2(recv_blocks), recvbuff, sendcount * send_blocks, sendtype, peer, 0, comm, &reqs[0]);
        CB_ILRECV(rank, CA_log2(recv_blocks), tmpbuff + (size / 2) * sbuff_size, recvcount * send_blocks, recvtype, peer, 0, comm, &reqs[1]);

        CB_LWAITALL(reqs, 2);

        memcpy(block, block_next, num_blocks * sizeof(unsigned int));

        mask <<= 1;
        inv_mask >>= 1;
        block_first_mask >>= 1;
    }

    for(int i = 0; i < size; i++) {
        int rot_i = 0;
        if(rank % 2 == 0) {
            rot_i = CA_mod(i - rank, size);
        }
        else {
            rot_i = CA_mod(rank - i, size);
        }

        int repr = 0;
        if(CA_in_bine_range(rot_i, CA_log2(size))) {
            repr = CA_rank2nb_raw(rot_i);
        }
        else {
            repr = CA_rank2nb_raw(rot_i - size);
        }

        int idx = CA_remap_ddbl(repr);

        int off_src = idx * sbuff_size;
        int off_dest = i * sbuff_size;

        memcpy((char*) recvbuff + off_dest, tmpbuff + off_src, sbuff_size);
    }

    free(tmpbuff);

    CB_COLL_END(comm, rank, 0, "out/butterfly/bine_alltoall.json");

    return MPI_SUCCESS;
}
