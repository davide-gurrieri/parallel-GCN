#ifndef GCN_H
#include "module.h"
#include "optim.h"
#include "sparse.h"
#include "variable.h"
#include <utility>
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
  std::vector<int> split;
  std::vector<int> label;
};

class GCN {
  std::vector<Module *> modules;
  std::vector<Variable> variables;
  // Variable *input, *output;
  Variable *output;
  std::vector<int> truth;
  Adam optimizer;
  float loss;
  void set_truth(int current_split);
  float get_accuracy();
  float get_l2_penalty();
  std::pair<float, float> train_epoch();
  std::pair<float, float> eval(int current_split);
  GCNData *data;

public:
  GCN(GCNParams params, GCNData *data);
  GCN();
  GCNParams params;
  ~GCN();
  void run();
};

#define GCN_H
#endif