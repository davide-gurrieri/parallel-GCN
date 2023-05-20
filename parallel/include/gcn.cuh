#ifndef GCN_CUH
#define GCN_CUH

#include "../include/module.cuh"
#include "../include/sparse.cuh"
#include "../include/optim.cuh"
#include "../include/shared_ptr.cuh"
#include "../include/utils.cuh"
#include "../include/variable.cuh"
#include "../include/reduction.cuh"
#include "../include/smart_object.cuh"

// #include <cuda_runtime.h>
#include <utility> // for std::pair
#include <memory>  // for std::shared_ptr and std::unique_ptr
#include <vector>

using std::shared_ptr;
using std::unique_ptr;

// ##################################################################################

struct GCNParams
{
  natural num_nodes, input_dim, hidden_dim{16}, output_dim;
  real dropout{0.5};
  natural epochs{100}, early_stopping{0};
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
  DevGCNData(const GCNData &gcn_data);
};

// ##################################################################################

class GCN
{
  const GCNData *data;
  DevGCNData dev_data;
  std::vector<unique_ptr<Module>> modules;
  std::vector<shared_ptr<Variable>> variables;
  shared_ptr<Variable> input, output;
  Adam optimizer;
  dev_shared_ptr<randState> dev_rand_states;
  dev_shared_ptr<integer> dev_truth;
  dev_shared_ptr<real> dev_l2;       // used by get_l2_penalty()
  dev_shared_ptr<natural> dev_wrong; // used by get_accuracy()

  void initialize_random();
  void initialize_truth();
  void set_input() const;
  void set_truth(const natural current_split) const;

  real get_accuracy() const;
  real get_l2_penalty() const;
  std::pair<real, real> train_epoch();
  std::pair<real, real> eval(const natural current_split) const;

public:
  // std::vector<smart_object<cudaStream_t>> streams;
  // std::vector<smart_object<cudaStream_t>> events;
  real loss;
  const GCNParams *params;
  const AdamParams *adam_params;
  GCN(GCNParams const *params_, AdamParams const *adam_params_, GCNData const *data_);
  void run();
};

// ##################################################################################

#endif