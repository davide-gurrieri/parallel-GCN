#include "../include/module.h"
#include "../include/optim.h"
#include "../include/parser.h"

using namespace std;


int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    if (argc < 2) {
        cout << "parallel_gcn graph_name [num_nodes input_dim hidden_dim output_dim"
                "dropout learning_rate, weight_decay epochs early_stopping]" << endl;
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
    }


    GCN gcn(params, &data); // Create and initialize and object of type GCN.
    gcn.run(); // Run the main function of the model in order to train and validate the solution.
    return EXIT_SUCCESS;
}