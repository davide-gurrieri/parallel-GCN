#include "../include/GetPot"
#include "../include/gcn.cuh"
#include "../include/optim.cuh"
#include "../include/parser.h"

#include <iostream>
#include <string>

int main(int argc, char **argv) {

  int dev;
  cudaDeviceProp devProp;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&devProp, dev);

  GCNParams params;
  AdamParams adam_params;

  params.hidden_dims = std::vector<natural>(1, 16);
  params.dropouts = std::vector<real>(2, 0.5);
  CudaParams::SEED = 19990304;
  CudaParams::N_BLOCKS = 5 * devProp.multiProcessorCount;
  CudaParams::N_THREADS = 128;

  const std::vector<std::string> input_names = {"citeseer", "cora", "pubmed"};

  std::ofstream file("./output/performance_gpu.txt");
  if (!file.is_open()) {
    std::cerr << "Could not open file" << std::endl;
    return EXIT_FAILURE;
  }

  for (const auto &input_name : input_names) {
    // Parse data
    GCNData data;
    Parser parser(&params, &data, input_name);
    if (!parser.parse()) {
      std::cerr << "Cannot read input: " << input_name << std::endl;
      exit(EXIT_FAILURE);
    }

    // GCN object creation
    GCN gcn(&params, &adam_params, &data);

    if (input_name == "citeseer") {
      CudaParams::N_BLOCKS = 5 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 128;
    } else if (input_name == "cora") {
      CudaParams::N_BLOCKS = 5 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 128;
    } else if (input_name == "pubmed") {
      CudaParams::N_BLOCKS = 5 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 128;
    }

    std::cout << std::endl;
    std::cout << "##################### " + input_name +
                     " #####################"
              << std::endl;

    // run the algorithm
    file << input_name << " ";
    const natural rep = 200;
    for (natural i = 0; i < rep; i++) {
      tmr_sum[0] = 0;
      gcn.run();
      file << gcn.output_for_tuning << " ";
    }
    file << std::endl;
  }

  Variable::dev_rand_states.~dev_shared_ptr();

  return EXIT_SUCCESS;
}