#include "CollBench/errors.h"
#include "CollBench/init.h"
#include "bench/bcast.h"
#include "bench/gather.h"
#include "bench/scatter.h"
#include <mpi.h>

int main(void) {
    CB_Error_t err = CB_SUCCESS;

    MPI_Init(NULL, NULL);
    CB_init();

    CB_CHECK(CA_bench_bine_bcast_dhlv(), cleanup);
    CB_CHECK(CA_bench_bine_gatherv(), cleanup);
    CB_CHECK(CA_bench_bine_gather(), cleanup);
    CB_CHECK(CA_bench_bine_scatter(), cleanup);
    CB_CHECK(CA_bench_bine_scatterv(), cleanup);

    cleanup:
        CB_finalize();
        MPI_Finalize();
        return err;
}
