#include "../include/GetPot"
#include "../include/gcn.cuh"
#include "../include/optim.cuh"
#include "../include/parser.h"

#include <iostream>
#include <string>

/*
n_layers = 2
hidden_dims = 16
dropouts = 0.5,0.5
epochs = 100
early_stopping = 0
weight_decay = 5e-4
*/

int main(int argc, char **argv) {

  int dev;
  cudaDeviceProp devProp;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&devProp, dev);

  /*
  const std::vector<std::string> input_names = {"citeseer", "cora", "pubmed"};
  const std::vector<real> dropout_values = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  const std::vector<natural> hidden_dim_values = {8, 16, 32, 64};
  const std::vector<natural> early_stopping_values = {10, 20};
  const std::vector<real> weight_decay_values = {5e-5, 5e-4, 5e-3, 5e-2};
  */

  const std::vector<std::string> input_names = {"citeseer", "cora", "pubmed"};
  const std::vector<real> dropout_values = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  const std::vector<natural> hidden_dim_values = {8, 16, 32, 64};
  const std::vector<natural> early_stopping_values = {10, 20};
  const std::vector<real> weight_decay_values = {5e-5, 5e-4, 5e-3, 5e-2};

  for (const auto &input_name : input_names) {
    GCNParams params;
    AdamParams adam_params;
    params.epochs = 1000;
    CudaParams::SEED = 19990304;

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
    std::ofstream file("./output/tuning_accuracy_" + input_name + ".txt");
    if (!file.is_open()) {
      std::cerr << "Could not open file" << std::endl;
      return EXIT_FAILURE;
    }
    // heading
    std::string heading = "early_stopping hidden_dim weight_decay dropout1 "
                          "dropout2 last_val_accuracy";
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
    for (const natural &early_stopping : early_stopping_values) {
      params.early_stopping = early_stopping;
      for (const natural &hidden_dim : hidden_dim_values) {
        params.hidden_dims = {hidden_dim};
        for (const real &weight_decay : weight_decay_values) {
          adam_params.weight_decay = weight_decay;
          for (const real &dropout1 : dropout_values) {
            for (const real &dropout2 : dropout_values) {
              params.dropouts = {dropout1, dropout2};

              // run the algorithm
              natural rep = 5;
              real sum = 0;
              for (natural i = 0; i < rep; i++) {
                // GCN object creation
                GCN gcn(&params, &adam_params, &data);
                reset_timer();
                gcn.run();
                sum += gcn.last_val_accuracy;
              }
              sum /= rep;
              std::string to_write = std::to_string(early_stopping) + " " +
                                     std::to_string(hidden_dim) + " " +
                                     std::to_string(weight_decay) + " " +
                                     std::to_string(dropout1) + " " +
                                     std::to_string(dropout2) + " " +
                                     std::to_string(sum);
              file << to_write << std::endl;
            }
          }
        }
      }
    }
    Variable::sizes.clear();
  }
  Variable::dev_rand_states.~dev_shared_ptr();

  return EXIT_SUCCESS;
}