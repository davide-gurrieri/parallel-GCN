#include "../include/rand.cuh"

__device__ real unif(RandState *state)
{
    uint64_t t = state->a;
    uint64_t const s = state->b;
    state->a = s;
    t ^= t << 23;
    t ^= t >> 17;
    t ^= s ^ (s >> 26);
    state->b = t;
    return static_cast<real>((t + s) & 0x7fffffff) / MY_RAND_MAX;
}
