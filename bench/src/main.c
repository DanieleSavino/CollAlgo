#include "CollBench/errors.h"
#include "CollBench/init.h"
#include "bench/bcast.h"
#include <mpi.h>

int main(void) {
    CB_Error_t err = CB_SUCCESS;

    MPI_Init(NULL, NULL);
    CB_init();

    CB_CHECK(CA_bench_bine_bcast_dhlv(), cleanup);

    cleanup:
        CB_finalize();
        MPI_Finalize();
        return err;
}
