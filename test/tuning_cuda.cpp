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

  const std::vector<std::string> input_names = {"citeseer", "cora", "pubmed",
                                                "reddit"};
  const std::vector<natural> threads = {128, 256, 512, 1024};

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

    std::ofstream file("./output/tuning_cuda_" + input_name + ".txt");

    if (!file.is_open()) {
      std::cerr << "Could not open file" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << std::endl;
    std::cout << "##################### " + input_name +
                     " #####################"
              << std::endl;
    for (natural num_blocks_factor = 1; num_blocks_factor <= 16;
         num_blocks_factor++) {
      for (const auto &num_threads : threads) {
        CudaParams::N_BLOCKS = num_blocks_factor * devProp.multiProcessorCount;
        CudaParams::N_THREADS = num_threads;

        // run the algorithm
        natural rep = (input_name == "reddit") ? 5 : 100;
        real sum = 0;
        for (natural i = 0; i < rep; i++) {
          tmr_sum[0] = 0;
          gcn.run();
          sum += gcn.avg_epoch_time;
        }
        sum /= rep;

        std::string to_write = std::to_string(num_blocks_factor) + " " +
                               std::to_string(num_threads) + " " +
                               std::to_string(sum);
        std::cout << "#####################" << std::endl;
        std::cout << std::to_string(num_blocks_factor) + " " +
                         std::to_string(num_threads)
                  << std::endl;
        std::cout << "#####################" << std::endl;

        file << to_write << std::endl;
      }
    }
    Variable::sizes.clear();
  }
  Variable::dev_rand_states.~dev_shared_ptr();

  return EXIT_SUCCESS;
}