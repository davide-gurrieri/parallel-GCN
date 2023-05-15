#ifndef GCN_CUH
#define GCN_CUH

#include <cuda_runtime.h>

#include <utility>

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
  natural train_dim{0}, val_dim{0}, test_dim{0};
  void print_info() const;
};

// ##################################################################################

struct GCNData
{
  SparseIndex feature_index, graph;
  std::vector<natural> split;
  std::vector<integer> label;
  std::vector<real> feature_value;
  std::vector<real> graph_value;
};

// ##################################################################################

class DevGCNData
{
public:
  DevSparseIndex dev_graph_index;   // adjacency matrix
  DevSparseIndex dev_feature_index; // feature
  dev_shared_ptr<real> dev_feature_value;
  dev_shared_ptr<real> dev_graph_value;
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

  void set_input();
  void set_truth(int current_split);

  real get_accuracy();
  real get_l2_penalty();
  std::pair<real, real> train_epoch();
  std::pair<real, real> eval(natural current_split);

public:
  real loss;
  DevGCNData dev_data;
  // ! dev_shared_ptr<curandState> dev_rand_states;
  dev_shared_ptr<randState> dev_rand_states;
  GCNParams *params;
  GCN(GCNParams *params_, GCNData *data_);
  void initialize_random();
  void initialize_truth();
  void run();
};

// ##################################################################################

#endif