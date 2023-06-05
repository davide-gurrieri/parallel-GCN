#ifndef TIMER_H
#include <chrono>
#include <vector>

typedef enum {
  TMR_TRAIN = 0,
  TMR_TEST,
  TMR_MATMUL_FW,
  TMR_MATMUL_BW,
  TMR_SPMATMUL_FW,
  TMR_SPMATMUL_BW,
  TMR_GRAPHSUM_FW,
  TMR_GRAPHSUM_BW,
  TMR_LOSS_FW,
  TMR_RELU_FW,
  TMR_RELU_BW,
  TMR_DROPOUT_FW,
  TMR_DROPOUT_BW,
  __NUM_TMR
} timer_instance;

void timer_start(timer_instance t);
float timer_stop(timer_instance t);
float timer_total(timer_instance t);
void reset_timer();

#define PRINT_TIMER_AVERAGE(T, E)                                              \
  printf(#T " average time: %.3fms\n", timer_total(T) * 1000 / E)
#define PRINT_TIMER_AVERAGE_EVAL(T, E)                                         \
  printf("%.3f\n", timer_total(T) * 1000 / E)

inline float TIMER_AVERAGE_NO_OUTPUT(timer_instance t, int e) {
  return timer_total(t) * 1000 / e;
}

#define TIMER_H
#endif
