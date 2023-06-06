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

  const std::vector<std::string> input_names = {"citeseer", "cora", "pubmed",
                                                "reddit"};

  std::ofstream file("./output/performance_gpu.txt");
  if (!file.is_open()) {
    std::cerr << "Could not open file" << std::endl;
    return EXIT_FAILURE;
  }

  for (const auto &input_name : input_names) {
    GCNParams params;
    AdamParams adam_params;

    // Parse data
    GCNData data;
    Parser parser(&params, &data, input_name);
    if (!parser.parse()) {
      std::cerr << "Cannot read input: " << input_name << std::endl;
      exit(EXIT_FAILURE);
    }

    if (input_name == "citeseer") {
      CudaParams::N_BLOCKS = 6 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 128;
    } else if (input_name == "cora") {
      CudaParams::N_BLOCKS = 2 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 1024;
    } else if (input_name == "pubmed") {
      CudaParams::N_BLOCKS = 8 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 256;
    } else if (input_name == "reddit") {
      CudaParams::N_BLOCKS = 16 * devProp.multiProcessorCount;
      CudaParams::N_THREADS = 512;
    }

    // GCN object creation
    GCN gcn(&params, &adam_params, &data);

    std::cout << std::endl;
    std::cout << "##################### " + input_name +
                     " #####################"
              << std::endl;

    // run the algorithm
    file << input_name << " ";
    const natural rep = (input_name == "reddit") ? 20 : 200;
    real sum = 0;
    real sum2 = 0;
    for (natural i = 0; i < rep; i++) {
      reset_timer();
      gcn.run();
      file << gcn.avg_epoch_time << " ";
      sum += gcn.total_time;
      sum2 += gcn.avg_epoch_time;
    }
    sum /= rep;
    sum2 /= rep;
    file << std::endl;
    std::cout << std::endl;
    std::cout << "Average total training + evaluation time of " << params.epochs
              << " epochs calculated over " << rep
              << " executions:" << std::endl;
    std::cout << sum << "s" << std::endl;
    std::cout << std::endl;
    std::cout << "Average epoch time calculated over " << rep << " x "
              << params.epochs << " executions:" << std::endl;
    std::cout << sum2 << "ms" << std::endl;
    std::cout << std::endl;

    Variable::sizes.clear();
  }
  Variable::dev_rand_states.~dev_shared_ptr();

  return EXIT_SUCCESS;
}