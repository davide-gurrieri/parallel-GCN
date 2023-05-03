#ifndef RAND_H
#include <assert.h>
#include <cstdint>
#include <cstdlib>


#define MY_RAND_MAX                                                            \
  0x7fffffff // is the maximum value that can be represented by a 32-bit signed
             // integer in hexadecimal notation. In decimal notation, it is
             // equal to 2,147,483,647.

void init_rand_state();
const int CACHE_LINE_SIZE = 64;
const int MAX_NUM_THREADS = 64;

uint32_t xorshift128plus(uint64_t *state);

extern uint64_t rand_state[2];
#define RAND() xorshift128plus(&rand_state[0])

#define RAND_H
#endif