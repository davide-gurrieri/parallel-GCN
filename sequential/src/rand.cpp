#include "../include/rand.h"
#include <immintrin.h>


uint64_t rand_state[2];

void init_rand_state() {
  int x = 0, y = 0;
  // returns a pseudo-random number in the range of 0 to RAND_MAX.
  // RAND_MAX is a constant whose default value may vary between implementations
  // but it is granted to be at least 32767
  while (x == 0 || y == 0) {
    x = rand();
    y = rand();
  }
  rand_state[0] = x;
  rand_state[1] = y;
}

uint32_t xorshift128plus(uint64_t *state) {
  uint64_t t = state[0];
  uint64_t const s = state[1];
  assert(t && s);
  state[0] = s;
  t ^= t << 23;       // a
  t ^= t >> 17;       // b
  t ^= s ^ (s >> 26); // c
  state[1] = t;
  uint32_t res = (t + s) & 0x7fffffff;
  return res;
}
