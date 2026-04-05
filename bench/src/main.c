#include "CollAlgo/utils.h"
#include "CollBench/errors.h"
#include "CollBench/init.h"
#include "bench/allgather.h"
#include "bench/alltoall.h"
#include "bench/bcast.h"
#include "bench/gather.h"
#include "bench/reduce.h"
#include "bench/scatter.h"
#include <mpi.h>

int main(void) {
    CB_Error_t err = CB_SUCCESS;

    MPI_Init(NULL, NULL);
    CB_CHECK(CB_init(), cleanup);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CA_root_print("Profiling bcast", rank, 0);
    CB_CHECK(CA_bench_bine_bcast_dhlv(), cleanup);

    CA_root_print("Profiling reduce", rank, 0);
    CB_CHECK(CA_bench_bine_reduce(), cleanup);

    CA_root_print("Profiling gatherv", rank, 0);
    CB_CHECK(CA_bench_bine_gatherv(), cleanup);

    CA_root_print("Profiling gather", rank, 0);
    CB_CHECK(CA_bench_bine_gather(), cleanup);

    CA_root_print("Profiling scatter", rank, 0);
    CB_CHECK(CA_bench_bine_scatter(), cleanup);

    CA_root_print("Profiling scatterv", rank, 0);
    CB_CHECK(CA_bench_bine_scatterv(), cleanup);

    CA_root_print("Profiling allgather", rank, 0);
    CB_CHECK(CA_bench_bine_allgather(), cleanup);

    CA_root_print("Profiling alltoall", rank, 0);
    CB_CHECK(CA_bench_bine_alltoall(), cleanup);

    CA_root_print("Profiling Done", rank, 0);

    cleanup:
        CB_CHECK_RET(CB_finalize());
        MPI_Finalize();
        return err;
}
