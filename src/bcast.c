#include "CollAlgo/bcast.h"
#include "CollAlgo/bine.h"
#include "CollAlgo/utils.h"
#include "CollBench/dist_list.h"
#include "CollBench/init.h"
#include <math.h>
#include <mpi.h>

int CA_bine_bcast_dhlv(void *buff, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    CB_COLL_START();

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(!CA_is_pow_2(size)) {
        return MPI_ERR_ASSERT;
    }

    int s = log2(size);
    int mask = 1 << (s - 1);

    int recvd = root == rank;

    int mod_rank = CA_mod(rank - root, size);
    int nb_rank = CA_rank2nb(mod_rank, s);

    while(mask > 0) {
        int mask_lsbs = (mask << 1) - 1;
        int nb_peer = nb_rank ^ mask_lsbs;
        int mod_peer = CA_nb2rank(nb_peer, s);
        int peer = CA_mod(mod_peer + root, size);

        if(recvd) {
            CB_LSEND(rank, s - __builtin_popcount(mask_lsbs), buff, count, datatype, peer, 0, comm);
        }
        else {
            int eq_lsbs = nb_rank & mask_lsbs;

            if(eq_lsbs == 0 || eq_lsbs == mask_lsbs) {
                CB_LRECV(rank, s - __builtin_popcount(mask_lsbs), buff, count, datatype, peer, 0, comm);
                recvd = 1;
            }
        }

        mask >>= 1;
    }

    CB_COLL_END(comm, root, "out/bine_bcast_dhlv.json");

    return MPI_SUCCESS;
}
