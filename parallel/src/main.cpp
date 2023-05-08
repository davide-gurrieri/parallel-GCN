//#include "../include/module.h"
//#include "../include/optim.h"
#include "../include/gcn.cuh"
#include "../include/parser.h"

int main(int argc, char **argv) {
  setbuf(stdout, NULL);
  if (argc < 2) {
    std::cout
        << "parallel_gcn graph_name [num_nodes input_dim hidden_dim output_dim"
           "dropout learning_rate, weight_decay epochs early_stopping]"
        << std::endl;
    return EXIT_FAILURE;
  }

  // Parse the selected dataset
  GCNParams params = GCNParams::get_default();
  GCNData data;
  std::string input_name(argv[1]);
  Parser parser(&params, &data, input_name);
  if (!parser.parse()) {
    std::cerr << "Cannot read input: " << input_name << std::endl;
    exit(EXIT_FAILURE);
  } else
    std::cout << "Parsing goes well." << std::endl;

  GCN cu_gcn(&params, &data);
  std::cout << "OK" << std::endl;

  return EXIT_SUCCESS;
}