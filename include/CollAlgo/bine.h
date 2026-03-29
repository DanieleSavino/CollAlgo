#include "CollAlgo/utils.h"
#include <assert.h>
#include <stdint.h>

#define BINE_MAX_STEPS 20

static int rhos[BINE_MAX_STEPS] = {
    1,   -1,    3,    -5,    11,    -21,    43,    -85,    171,    -341,
    683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};

static int smallest_negabinary[BINE_MAX_STEPS] = {
    0,    0,    -2,    -2,    -10,    -10,    -42,    -42,    -170,    -170,
    -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[BINE_MAX_STEPS] = {
    0,   1,    1,    5,    5,    21,    21,    85,    85,    341,
    341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

static inline int CA_rank2nb(int32_t rank, int bits) {
  const int size = (1 << bits);
  if (rank > 0x55555555)
    return -1;

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

static inline int CA_rank2nb_raw(int32_t rank) {
  if (rank > 0x55555555)
    return -1;

  const uint32_t mask = 0xAAAAAAAA;
  const int32_t val = (mask + rank) ^ mask;

  return val;
}

static inline int CA_nb2rank_raw(int32_t nb) {
  const uint32_t mask = 0xAAAAAAAA;
  const int32_t val = (mask ^ nb) - mask;
  return val;
}

static inline int CA_pi(int rank, int step, int comm_sz) {
  int dest;

  if ((rank & 1) == 0)
    dest = (rank + rhos[step]) % comm_sz; // Even rank
  else
    dest = (rank - rhos[step]) % comm_sz; // Odd rank

  if (dest < 0)
    dest += comm_sz; // Adjust for negative ranks

  return dest;
}

static inline int CA_in_bine_range(int x, uint32_t nbits){
    return x >= smallest_negabinary[nbits] && x <= largest_negabinary[nbits];
}

static inline uint32_t CA_reverse32(uint32_t x){
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}

static inline int CA_log2(int value) {
  int log = sizeof(int) * 8 - 1 - __builtin_clz(value);
  if (!CA_is_pow_2(value)) {
    log += 1;
  }
  return log;
}

static inline uint32_t CA_nb_to_nu(uint32_t nb, uint32_t size) {
  return CA_reverse32(nb ^ (nb >> 1)) >> (32 - CA_log2(size));
}

static inline uint32_t CA_nu(uint32_t rank, uint32_t size) {
  uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
  size_t num_bits = CA_log2(size);
  if (rank % 2) {
    if (CA_in_bine_range(rank, num_bits)) {
      nba = CA_rank2nb_raw(rank);
    }
    if (CA_in_bine_range(rank - size, num_bits)) {
      nbb = CA_rank2nb_raw(rank - size);
    }
  } else {
    if (CA_in_bine_range(-rank, num_bits)) {
      nba = CA_rank2nb_raw(-rank);
    }
    if (CA_in_bine_range(-rank + size, num_bits)) {
      nbb = CA_rank2nb_raw(-rank + size);
    }
  }
  assert(nba != UINT32_MAX || nbb != UINT32_MAX);

  if (nba == UINT32_MAX && nbb != UINT32_MAX) {
    return CA_nb_to_nu(nbb, size);
  } else if (nba != UINT32_MAX && nbb == UINT32_MAX) {
    return CA_nb_to_nu(nba, size);
  } else { // Check MSB
    int nu_a = CA_nb_to_nu(nba, size);
    int nu_b = CA_nb_to_nu(nbb, size);
    if (nu_a < nu_b) {
      return nu_a;
    } else {
      return nu_b;
    }
  }
}
