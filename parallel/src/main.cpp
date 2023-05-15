//#include "../include/module.h"
//#include "../include/optim.h"
#include "../include/GetPot"
#include "../include/gcn.cuh"
#include "../include/parser.h"

int
main(int argc, char** argv)
{
  setbuf(stdout, NULL);
  if (argc < 2) {
    std::cout
      << "parallel_gcn graph_name [num_nodes input_dim hidden_dim output_dim"
         "dropout learning_rate, weight_decay epochs early_stopping]"
      << std::endl;
    return EXIT_FAILURE;
  }

  GetPot command_line(argc, argv);
  const std::string file_name = command_line("file", "./parameters.txt");
  GetPot datafile(file_name.c_str());
  GCNParams params;
  params.hidden_dim = datafile("hidden_dim", 1);
  params.dropout = datafile("dropout", 0.0);
  params.learning_rate = datafile("learning_rate", 0.0);
  params.weight_decay = datafile("weight_decay", 0.0);
  params.epochs = datafile("epochs", 0);
  params.early_stopping = datafile("early_stopping", 0);

  print_gpu_info();
  // Parse the selected dataset
  // GCNParams params = GCNParams::get_default();
  GCNData data;
  std::string input_name(argv[1]);
  Parser parser(&params, &data, input_name);
  if (!parser.parse()) {
    std::cerr << "Cannot read input: " << input_name << std::endl;
    exit(EXIT_FAILURE);
  } else
    std::cout << "Parsing goes well." << std::endl;
  params.print_info();
  GCN gcn(&params, &data);
  gcn.run();

  return EXIT_SUCCESS;
}