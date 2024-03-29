#include "../include/GetPot"
#include "../include/gcn.cuh"
#include "../include/optim.cuh"
#include "../include/parser.h"

#include <cmath>
#include <iostream>
#include <random>
#include <string>

/*
n_layers = 2
hidden_dims = 16
dropouts = 0.5,0.5
epochs = 100
early_stopping = 0
weight_decay = 5e-4
*/

real sd(const std::vector<real> &data, const real mean) {
  real sumSquaredDeviations = 0.0;
  for (const auto &element : data) {
    real deviation = element - mean;
    sumSquaredDeviations += deviation * deviation;
  }
  return sqrt(sumSquaredDeviations / (data.size() - 1));
}

int main(int argc, char **argv) {

  int dev;
  cudaDeviceProp devProp;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&devProp, dev);

  std::mt19937 gen;
  std::uniform_int_distribution<> distr(0, RAND_MAX);

  /*
  std::vector<natural> n_layers_values = {2, 3, 4};           // 3
  std::vector<real> dropout_values = {0.0, 0.2, 0.4, 0.6};    // 16
  std::vector<natural> hidden_dim_values = {8, 16, 32, 64};   // 4
  std::vector<natural> early_stopping_values = {10};          // 1
  std::vector<real> weight_decay_values = {5e-5, 5e-4, 5e-3}; // 3
  */
  std::vector<std::string> input_names = {"citeseer", "cora", "pubmed"};

#ifdef SECOND
#ifdef NO_FEATURE
  input_names = {"cora", "pubmed"};
#else
  input_names = {"citeseer"};
#endif
#endif

  std::vector<natural> n_layers_values = {2, 3, 4};
  std::vector<real> dropout_values = {0.0, 0.2, 0.4, 0.6};
  std::vector<natural> hidden_dim_values = {8, 16, 32, 64};
  std::vector<natural> early_stopping_values = {10};
  std::vector<real> weight_decay_values = {5e-5, 5e-4, 5e-3};

  for (const auto &input_name : input_names) {
#ifdef SECOND
    if (input_name == "citeseer") {
      n_layers_values = {2};
      dropout_values = {0.2, 0.4, 0.6, 0.8};
      hidden_dim_values = {12, 20, 40};
      early_stopping_values = {10};
      weight_decay_values = {5e-5, 5e-4};
    } else if (input_name == "cora") {
      n_layers_values = {2};
      dropout_values = {0.0, 0.2, 0.4};
      hidden_dim_values = {56, 72, 80};
      early_stopping_values = {10};
      weight_decay_values = {5e-5, 5e-4};
    } else if (input_name == "pubmed") {
      n_layers_values = {2};
      dropout_values = {0.0, 0.2, 0.4};
      hidden_dim_values = {4, 12, 20};
      early_stopping_values = {10};
      weight_decay_values = {5e-4, 5e-3};
    }
#endif
    const natural total_rep = n_layers_values.size() * dropout_values.size() *
                              dropout_values.size() * hidden_dim_values.size() *
                              early_stopping_values.size() *
                              weight_decay_values.size();
    GCNParams params;
    AdamParams adam_params;
    params.epochs = 1000;

    std::cout << std::endl;
    std::cout << "##################### " + input_name +
                     " #####################"
              << std::endl;
    // Parse data
    GCNData data;
    Parser parser(&params, &data, input_name);
    if (!parser.parse()) {
      std::cerr << "Cannot read input: " << input_name << std::endl;
      exit(EXIT_FAILURE);
    }
// Open output file
#ifdef SECOND
    std::ofstream file("./output/tuning_accuracy_second_" + input_name +
                       ".txt");
#else
#ifndef NO_FEATURE
    std::ofstream file("./output/zztuning_accuracy_" + input_name + ".txt");
#else
    std::ofstream file("./output/zztuning_accuracy_no_feature_" + input_name +
                       ".txt");
#endif
#endif
    if (!file.is_open()) {
      std::cerr << "Could not open file" << std::endl;
      return EXIT_FAILURE;
    }
    // heading
    std::string heading =
        "n_layers early_stopping hidden_dim weight_decay dropout1 "
        "dropout2 mean_last_val_accuracy max_last_val_accuracy variance seed";
    file << heading << std::endl;

    // Set optimal cuda parameters
    if (input_name == "citeseer") {
      CudaParams::N_BLOCKS = 6 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 128;
    } else if (input_name == "cora") {
      CudaParams::N_BLOCKS = 6 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 1024;
    } else if (input_name == "pubmed") {
      CudaParams::N_BLOCKS = 8 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 256;
    }
    // Explore parameters values
    natural rep_number = 0;
    for (const auto &n_layers : n_layers_values) {
      params.n_layers = n_layers;
      for (const natural &early_stopping : early_stopping_values) {
        params.early_stopping = early_stopping;
        for (const natural &hidden_dim : hidden_dim_values) {
          params.hidden_dims = std::vector<natural>(n_layers - 1, hidden_dim);
          for (const real &weight_decay : weight_decay_values) {
            adam_params.weight_decay = weight_decay;
            for (const real &dropout1 : dropout_values) {
              for (const real &dropout2 : dropout_values) {
                rep_number++;
                params.dropouts.clear();
                params.dropouts.push_back(dropout1);
                std::vector<real> rep_dropouts(n_layers - 1, dropout2);
                params.dropouts.insert(params.dropouts.end(),
                                       rep_dropouts.begin(),
                                       rep_dropouts.end());
                // run the algorithm
                natural rep = 20;
                std::vector<real> val_accuracy_values(rep);
                real max_acc = 0;
                int seed;
                real mean = 0;
                for (natural i = 0; i < rep; i++) {
                  // GCN object creation
                  CudaParams::SEED = distr(gen);
                  GCN gcn{&params, &adam_params, &data};
                  reset_timer();
                  gcn.run();
                  const real val_acc = gcn.last_val_accuracy * 100;
                  if (val_acc > max_acc) {
                    max_acc = val_acc;
                    seed = CudaParams::SEED;
                  }
                  val_accuracy_values[i] = val_acc;
                  mean += val_acc;
                }
                mean /= rep;
                const real var = sd(val_accuracy_values, mean);
                std::string to_write =
                    std::to_string(n_layers) + " " +
                    std::to_string(early_stopping) + " " +
                    std::to_string(hidden_dim) + " " +
                    std::to_string(weight_decay) + " " +
                    std::to_string(dropout1) + " " + std::to_string(dropout2) +
                    " " + std::to_string(mean) + " " + std::to_string(max_acc) +
                    " " + std::to_string(var) + " " + std::to_string(seed);
                file << to_write << std::endl;
                printf("%.1f%%\n",
                       static_cast<real>(rep_number) / total_rep * 100);
                // std::cout << static_cast<real>(rep_number) / total_rep * 100
                //<< "%" << std::endl;
              }
            }
          }
        }
        Variable::sizes.clear();
      }
    }
  }
  Variable::dev_rand_states.~dev_shared_ptr();

  return EXIT_SUCCESS;
}