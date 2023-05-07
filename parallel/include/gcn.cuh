#ifndef GCN_CUH
#define GCN_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
// #include <memory>
#include "../include/variable.cuh"
#include "../include/module.cuh"
#include "../include/sparse.cuh"

// using std::shared_ptr;
// using std::unique_ptr;

/*
#include "module.h"
#include "optim.h"
#include "variable.h"
#include <utility>
*/

// #include <cstdio>
// #include <tuple>
// #include <vector>

struct GCNParams
{
  natural num_nodes, input_dim, hidden_dim, output_dim;
  real dropout, learning_rate, weight_decay;
  natural epochs, early_stopping;
  static GCNParams get_default();
};

struct GCNData
{
  SparseIndex feature_index, graph;
  std::vector<natural> split;
  std::vector<natural> label;
  std::vector<real> feature_value;
};

class DevGCNData
{
public:
  DevSparseIndex dev_graph;         // adjacency matrix
  DevSparseIndex dev_feature_index; // feature
  real *dev_feature_value;
  natural *dev_split;
  natural *dev_label;
  natural label_size;
  // DevGCNData();
  DevGCNData(const GCNData &gcn_data);

  ~DevGCNData();
};

class GCN
{
  GCNData *data;
  std::vector<Module *> modules;
  std::vector<Variable *> variables;
  Variable *input, *output;
  // Adam *optimizer;
  // integer *truth;
  real loss;
  /*
    void set_input();
    void set_truth(int current_split);
    float get_accuracy();
    float get_l2_penalty();
    pair<float, float> train_epoch();
    pair<float, float> eval(int current_split);
  */
public:
  inline static natural count = 0;
  DevGCNData dev_data;
  curandState *dev_rand_states;
  GCNParams *params;
  GCN(GCNParams *params_, GCNData *data_);
  void initialize_random();
  ~GCN();
  // void run();
};

#endif