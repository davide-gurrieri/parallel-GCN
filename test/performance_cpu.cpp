#include "../hpdga-spring23/include/module.h"
#include "../hpdga-spring23/include/optim.h"
#include "../hpdga-spring23/include/parser.h"
#include "../hpdga-spring23/include/timer.h"

using namespace std;

int main(int argc, char **argv) {
  setbuf(stdout, NULL);

  GCNParams params = GCNParams::get_default();
  GCNData data;

  const std::vector<std::string> input_names = {"citeseer", "cora", "pubmed",
                                                "reddit"};

  std::ofstream file("./output/performance_cpu.txt");
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
    GCN gcn(params, &data); // Create and initialize and object of type GCN.

    std::cout << std::endl;
    std::cout << "##################### " + input_name +
                     " #####################"
              << std::endl;

    // run the algorithm
    file << input_name << " ";
    const unsigned rep = (input_name == "reddit") ? 5 : 50;
    for (unsigned i = 0; i < rep; i++) {
      reset_timer();
      gcn.run();
      file << std::to_string(gcn.time) << " ";
    }
    file << std::endl;
  }

  return EXIT_SUCCESS;
}