#include "CollAlgo/reduce.h"
#include "CollAlgo/utils.h"
#include "CollAlgo/bine.h"
#include "CollBench/dist_list.h"
#include "CollBench/init.h"
#include <mpi.h>
#include <stddef.h>

#ifdef  CA_CUDA
    #include "CollAlgo/cuda/reduce.h"
#endif

int local_reduce(const void *src, void *dst, int count, MPI_Datatype dt, MPI_Op op, int cuda) {

#ifdef CA_CUDA
    if(cuda) {
        CA_cuda_reduce(src, dst, count, dt, op, 32);
        return 0;
    }
#else
    (void)cuda;
#endif

    CA_MPI_CHECK(MPI_Reduce_local(src, dst, count, dt, op));

    return 0;
}

int CA_bine_reduce(const void *sendbuff, void *recvbuff, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {

#ifdef CA_CUDA
    int use_cuda = CA_is_devptr(sendbuff);
#else
    int use_cuda = 0;
#endif

    int rank, size, dtsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Type_size(datatype, &dtsize);

    if(!CA_is_pow_2(size)) {
        return MPI_ERR_ASSERT;
    }

    CB_COLL_START();

    char *tmpbuff;
    CA_UMALLOC(tmpbuff, count * dtsize, use_cuda);

    if(rank != root) {
        CA_UMALLOC(recvbuff, count * dtsize, use_cuda);
    }

    CA_UMEMCPY(recvbuff, sendbuff, count * dtsize, use_cuda);

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
            local_reduce(tmpbuff, recvbuff, count, datatype, op, use_cuda);
        }
        mask <<= 1;
    }

    CA_UFREE(tmpbuff, use_cuda);
    if(rank != root) {
        CA_UFREE(recvbuff, use_cuda);
    }

    CB_COLL_END(comm, rank, root, use_cuda ? "out/tree/bine_reduce_cuda.json" : "out/tree/bine_reduce.json");
    return MPI_SUCCESS;
}
