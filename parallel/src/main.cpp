#ifdef DYNAMIC_INPUT
#include "../include/GetPot"
#endif

#include "../include/gcn.cuh"
#include "../include/optim.cuh"
#include "../include/parser.h"

#include <iostream>
#include <string>

int main(int argc, char **argv) {

  // setbuf(stdout, NULL);
  //  Print device informations
  natural multiProcessorCount = print_gpu_info();

  if (argc < 2) {
    std::cerr
        << "Give one input file name as argument [cora pubmed citeseer reddit]"
        << std::endl;
    return EXIT_FAILURE;
  }

  std::string input_name(argv[1]);

#ifdef DYNAMIC_INPUT
  // Read parameters at runtime from "parameters.txt" using GetPot
  GCNParams params;
  AdamParams adam_params;
  GetPot command_line(argc, argv);
#ifdef TUNE
  const std::string file_name = command_line("file", "./parameters.txt");
#else

  const std::string name =
      "./specific_parameters/parameters_" + input_name + ".txt";
  std::cout << name << std::endl;
  const std::string file_name = command_line("file", name.c_str());

#endif
  GetPot datafile(file_name.c_str());
  // GCNParams
  params.hidden_dim = datafile("hidden_dim", 0);
  params.dropout_input = datafile("dropout_input", 0.0);
  params.dropout_layer1 = datafile("dropout_layer1", 0.0);
  params.epochs = datafile("epochs", 0);
  params.early_stopping = datafile("early_stopping", 0);
  // AdamParams
  adam_params.learning_rate = datafile("learning_rate", 0.0);
  adam_params.weight_decay = datafile("weight_decay", 0.0);
  adam_params.beta1 = datafile("beta1", 0.0);
  adam_params.beta2 = datafile("beta2", 0.0);
  adam_params.eps = datafile("eps", 0.0);
  // CudaParams
  N_BLOCKS = datafile("num_blocks_factor", 0) * multiProcessorCount;
  N_THREADS = datafile("num_threads", 0);

#else
  GCNParams params;
  constexpr AdamParams adam_params;
  N_BLOCKS = 4 * multiProcessorCount;
  N_THREADS = 256;
#endif

  // Parse data
  GCNData data;
  Parser parser(&params, &data, input_name);
  if (!parser.parse()) {
    std::cerr << "Cannot read input: " << input_name << std::endl;
    exit(EXIT_FAILURE);
  }

// print parsed parameters
#ifndef TUNE
  params.print_info();
#endif

  // GCN object creation
  GCN gcn(&params, &adam_params, &data);

  // run the algorithm
  gcn.run();

  return EXIT_SUCCESS;
}