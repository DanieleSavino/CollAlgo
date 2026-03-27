#include <stdint.h>

static inline int CA_rank2nb(int32_t rank, int bits) {
    const int size = (1 << bits);
    if(rank > 0x55555555) return -1;

    const uint32_t mask = 0xAAAAAAAA;
    const int32_t val = (mask + rank) ^ mask;

    return val & (size - 1);
}

static inline int CA_nb2rank(int32_t nb, int bits) {
    const int size = (1 << bits);
    const uint32_t mask = 0xAAAAAAAA;
    const int32_t val = (mask ^ nb) - mask;
    return val & (size - 1);
}

static inline int CA_mod(int a, int b){
    int r = a % b;
    return r < 0 ? r + b : r;
}

static inline int CA_is_pow_2(int num) {
    return num > 0 && (num & (num - 1)) == 0;
}
