#include "CollAlgo/reduce.h"
#include "CollAlgo/utils.h"
#include "CollAlgo/bine.h"
#include "CollBench/dist_list.h"
#include "CollBench/init.h"
#include <mpi.h>
#include <string.h>

int CA_bine_reduce(const void *sendbuff, void *recvbuff, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    CB_COLL_START();

    int rank, size, dtsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Type_size(datatype, &dtsize);

    if(!CA_is_pow_2(size)) {
        return MPI_ERR_ASSERT;
    }

    char *tmpbuff;
    CA_MALLOC(tmpbuff, count * dtsize);

    if(rank != root) {
        CA_MALLOC(recvbuff, count * dtsize);
    }

    memcpy(recvbuff, sendbuff, count * dtsize);

    int s = CA_log2(size);

    int mod_rank = CA_mod(rank - root, size);
    int nb_rank = CA_rank2nb(mod_rank, s);
    int mask = 0x1;

    while(mask < size) {
        int mask_peer = (mask << 1) - 1;
        int nb_peer = nb_rank ^ mask_peer;
        int mod_peer = CA_nb2rank(nb_peer, s);
        int peer = CA_mod(mod_peer + root, size);

        int mask_lsbs = (mask << 2) - 1;
        int lsbs = nb_rank & mask_lsbs;
        int eq_lsbs = lsbs == 0 || lsbs == mask_lsbs;

        if(!eq_lsbs || ((mask << 1) >= size && (rank != root))) {
            CB_LSEND(rank, __builtin_popcount(mask_peer), recvbuff, count, datatype, peer, 0, comm);
            break;
        }
        else {
            CB_LRECV(rank, __builtin_popcount(mask_peer), tmpbuff, count, datatype, peer, 0, comm);
            CA_MPI_CHECK(MPI_Reduce_local(tmpbuff, recvbuff, count, datatype, op), cleanup);
        }
        mask <<= 1;
    }

    free(tmpbuff);
    if(rank != root) {
        free(recvbuff);
    }

    CB_COLL_END(comm, rank, root, "out/tree/bine_reduce.json");
    return MPI_SUCCESS;
}
