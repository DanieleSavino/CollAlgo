#include "CollBench/errors.h"
#include "CollBench/init.h"
#include "bench/allgather.h"
#include "bench/bcast.h"
#include "bench/gather.h"
#include "bench/scatter.h"
#include <mpi.h>
#include <stdio.h>

int main(void) {
    CB_Error_t err = CB_SUCCESS;

    MPI_Init(NULL, NULL);
    CB_init();

    printf("Profiling bcast\n");
    CB_CHECK(CA_bench_bine_bcast_dhlv(), cleanup);

    printf("Profiling gatherv\n");
    CB_CHECK(CA_bench_bine_gatherv(), cleanup);

    printf("Profiling gather\n");
    CB_CHECK(CA_bench_bine_gather(), cleanup);

    printf("Profiling scatter\n");
    CB_CHECK(CA_bench_bine_scatter(), cleanup);

    printf("Profiling scatterv\n");
    CB_CHECK(CA_bench_bine_scatterv(), cleanup);

    printf("Profiling allgather\n");
    CB_CHECK(CA_bench_bine_allgather(), cleanup);

    printf("Profiling Done\n");

    cleanup:
        CB_finalize();
        MPI_Finalize();
        return err;
}
