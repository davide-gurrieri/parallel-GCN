#ifndef GCN_CUH
#define GCN_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../include/variable.cuh"
#include "../include/module.cuh"
#include "../include/sparse.cuh"
#include "../include/optim.cuh"
#include "../include/shared_ptr.cuh"

// ##################################################################################

struct GCNParams
{
  natural num_nodes, input_dim, hidden_dim, output_dim;
  real dropout, learning_rate, weight_decay;
  natural epochs, early_stopping;
  static GCNParams get_default();
};

// ##################################################################################

struct GCNData
{
  SparseIndex feature_index, graph;
  std::vector<natural> split;
  std::vector<integer> label;
  std::vector<real> feature_value;
};

// ##################################################################################

class DevGCNData
{
public:
  DevSparseIndex dev_graph;         // adjacency matrix
  DevSparseIndex dev_feature_index; // feature
  dev_shared_ptr<real> dev_feature_value;
  dev_shared_ptr<natural> dev_split;
  dev_shared_ptr<integer> dev_label;
  natural label_size;
  // DevGCNData();
  DevGCNData(const GCNData &gcn_data);
};

// ##################################################################################

class GCN
{
  GCNData *data;
  std::vector<unique_ptr<Module>> modules;
  std::vector<shared_ptr<Variable>> variables;
  shared_ptr<Variable> input, output;
  unique_ptr<Adam> optimizer;
  dev_shared_ptr<integer> dev_truth;
  real loss;

  void set_input();
  void set_truth(int current_split);
  /*
    float get_accuracy();
    float get_l2_penalty();
    pair<float, float> train_epoch();
    pair<float, float> eval(int current_split);
  */
public:
  DevGCNData dev_data;
  dev_shared_ptr<curandState> dev_rand_states;
  GCNParams *params;
  GCN(GCNParams *params_, GCNData *data_);
  void initialize_random();
  void initialize_truth();
  // void run();
};

// ##################################################################################

#endif