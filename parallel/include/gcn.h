#ifndef GCN_H
/*
#include "module.h"
#include "optim.h"
#include "variable.h"
#include <utility>
*/
#include "sparse.h"
#include <vector>

struct GCNParams {
  int num_nodes, input_dim, hidden_dim, output_dim;
  float dropout, learning_rate, weight_decay;
  int epochs, early_stopping;
  static GCNParams get_default();
};

class GCNData {
public:
  SparseIndex feature_index, graph;
  std::vector<float> feature_value;
  std::vector<int> split;
  std::vector<int> label;
};

#define GCN_H
#endif