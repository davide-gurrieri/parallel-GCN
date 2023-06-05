#include "../include/timer.h"

void timer_start(timer_instance t) {
  tmr_t0[t] = std::chrono::high_resolution_clock::now();
}

float timer_stop(timer_instance t) {
  float count = std::chrono::duration_cast<std::chrono::duration<float>>(
                    std::chrono::high_resolution_clock::now() - tmr_t0[t])
                    .count();
  tmr_sum[t] += count;
  return count;
}

float timer_total(timer_instance t) { return tmr_sum[t]; }

void reset_timer() {
  for (int i = 0; i < __NUM_TMR; i++)
    tmr_sum[i] = 0.0f;
}
