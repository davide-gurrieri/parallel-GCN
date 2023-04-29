#include "../include/gcn.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include <cstdio>
#include <tuple>

/**
 * Returns the default paramets of the model
 * they will be overwritten by the parser when reading the dataset
*/
GCNParams GCNParams::get_default() {
    /*
    return { // citeseer
        3327,   // num_nodes
        3703,   // input_dim
        16,     // hidden_dim
        6,      // output_dim
        0.5,    // dropouyt
        0.01,   // learning_rate
        5e-4,   // weight_decay
        100,    // epochs
        0};     // early_stopping
    
    */

    ///*
    return { // CORA
        2708,   // num_nodes
        1433,   // input_dim
        16,     // hidden_dim
        7,      // output_dim
        0.5,    // dropouyt
        0.01,   // learning_rate
        5e-4,   // weight_decay
        100,    // epochs
        0};     // early_stopping
    //*/

    /*return { // PUBMED
        19717,   // num_nodes
        500,   // input_dim
        16,     // hidden_dim
        3,      // output_dim
        0.5,    // dropouyt
        0.01,   // learning_rate
        5e-4,   // weight_decay
        100,    // epochs
        0};     // early_stopping*/
}

GCN::GCN(GCNParams params, GCNData *input_data) {
    init_rand_state();
    this->params = params;
    data = input_data;
    modules.reserve(8); // allocate the space for the 8 modules/layers
    variables.reserve(8);
    variables.emplace_back(data->feature_index.indices.size(), false);
    input = &variables.back();

    // dropout
    modules.push_back(new Dropout(input, params.dropout));
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var1 = &variables.back();
    variables.emplace_back(params.input_dim * params.hidden_dim, true, true);
    Variable *layer1_weight = &variables.back();
    layer1_weight->glorot(params.input_dim, params.hidden_dim); // weights initilization
    
    // sparsematmul
    modules.push_back(new SparseMatmul(input, layer1_weight, layer1_var1, &data->feature_index, params.num_nodes, params.input_dim, params.hidden_dim));
    variables.emplace_back(params.num_nodes * params.hidden_dim);
    Variable *layer1_var2 = &variables.back();
    
    // graphsum
    modules.push_back(new GraphSum(layer1_var1, layer1_var2, &data->graph, params.hidden_dim));
    
    // RELU
    modules.push_back(new ReLU(layer1_var2));
    
    // dropout
    modules.push_back(new Dropout(layer1_var2, params.dropout));
    variables.emplace_back(params.num_nodes * params.output_dim);
    Variable *layer2_var1 = &variables.back();
    variables.emplace_back(params.hidden_dim * params.output_dim, true, true);
    Variable *layer2_weight = &variables.back();
    layer2_weight->glorot(params.hidden_dim, params.output_dim); // weights initilization
    
    // dense matrix multiply
    modules.push_back(new Matmul(layer1_var2, layer2_weight, layer2_var1, params.num_nodes, params.hidden_dim, params.output_dim));
    variables.emplace_back(params.num_nodes * params.output_dim);
    output = &variables.back();
    
    // graph sum
    modules.push_back(new GraphSum(layer2_var1, output, &data->graph, params.output_dim));
    truth = std::vector<int>(params.num_nodes);
    
    // cross entropy loss
    modules.push_back(new CrossEntropyLoss(output, truth.data(), &loss, params.output_dim));
    
    // Adam optimization algorithm (alternative to the classical stochastic gradient descent)
    AdamParams adam_params = AdamParams::get_default();
    adam_params.lr = params.learning_rate;
    adam_params.weight_decay = params.weight_decay;
    optimizer = Adam({{layer1_weight, true}, {layer2_weight, false}}, adam_params);
}

GCN::~GCN(){
    for(auto m: modules)
        delete m;
}

// set the current input for the GCN model
void GCN::set_input() {
    for (int i = 0; i < input->data.size(); i++)
        input->data[i] = data->feature_value[i];
}

// set the label of each node inside of the current_split (validation/train/test)
void GCN::set_truth(int current_split) {
    for(int i = 0; i < params.num_nodes; i++)
        // truth[i] is the real label of "i"
        truth[i] = data->split[i] == current_split ? data->label[i] : -1;
}

// get the current accuracy of the model
float GCN::get_accuracy() {
    int wrong = 0, total = 0;
    for(int i = 0; i < params.num_nodes; i++) {
        if(truth[i] < 0) continue;
        total++;
        float truth_logit = output->data[i * params.output_dim + truth[i]];
        for(int j = 0; j < params.output_dim; j++)
            if (output->data[i * params.output_dim + j] > truth_logit) {
                wrong++;
                break;
            }
    }
    return float(total - wrong) / total;
}

// reduce the likelihood of model overfitting by using l2 regularization
float GCN::get_l2_penalty() {
    float l2 = 0;
    for (int i = 0; i < variables[2].data.size(); i++) {
        float x = variables[2].data[i];
        l2 += x * x;
    }
    return params.weight_decay * l2 / 2;
}

/**
 * Train an epoch of the model
*/
std::pair<float, float> GCN::train_epoch() {
    set_input(); // set the input data

    set_truth(1); // get the true labels for the dataset with split == 1 (train)

    for (auto m: modules) // iterate over the layer applying a forward pass
        m->forward(true);

    float train_loss = loss + get_l2_penalty(); // correct the loss with the l2 regularization
    float train_acc = get_accuracy(); // compute the accuracy comparing the prediction against the truth
    for (int i = modules.size() - 1; i >= 0; i--) // do a backward pass on the layers
        modules[i]->backward();

    optimizer.step(); // apply a step of the adapm optimization

    return {train_loss, train_acc};
}

/**
 * current_split == 2 --> validation
 * current_split == 3 --> test
*/
std::pair<float, float> GCN::eval(int current_split) {
    set_input();
    set_truth(current_split);
    for (auto m: modules)
        m->forward(false);
    float test_loss = loss + get_l2_penalty();
    float test_acc = get_accuracy();
    return {test_loss, test_acc};
}

void GCN::run() {
    int epoch = 1;
    float total_time = 0.0;
    std::vector<float> loss_history;
    // Iterate the training process based on the selected number of epoch
    for(; epoch <= params.epochs; epoch++) {
        float train_loss, train_acc, val_loss, val_acc;
        timer_start(TMR_TRAIN); // just for timing purposes
        std::tie(train_loss, train_acc) = train_epoch(); // train the epoch and record the current train_loss and train_accuracy
        std::tie(val_loss, val_acc) = eval(2); //eval the model at the current step in order to obtain the val_loss and val_accuracy
        printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f time=%.5f\n",
            epoch, train_loss, train_acc, val_loss, val_acc, timer_stop(TMR_TRAIN));

        loss_history.push_back(val_loss); // record the validation loss in order to apply an early stopping mechanism

        //early stopping mechanism
        if(params.early_stopping > 0 && epoch >= params.early_stopping) {
            float recent_loss = 0.0;
            for(int i = epoch - params.early_stopping; i < epoch; i++)
                recent_loss += loss_history[i];
            if (val_loss > recent_loss / params.early_stopping) {
                printf("Early stopping...\n");
                break;
            }
        }
    }
    PRINT_TIMER_AVERAGE(TMR_TRAIN, epoch);
    PRINT_TIMER_AVERAGE(TMR_MATMUL_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_MATMUL_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_SPMATMUL_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_SPMATMUL_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_GRAPHSUM_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_GRAPHSUM_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_RELU_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_RELU_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_DROPOUT_FW, epoch);
    PRINT_TIMER_AVERAGE(TMR_DROPOUT_BW, epoch);
    PRINT_TIMER_AVERAGE(TMR_LOSS_FW, epoch);
    

    float test_loss, test_acc;
    timer_start(TMR_TEST);
    std::tie(test_loss, test_acc) = eval(3); // eval the model on the test set
    printf("test_loss=%.5f test_acc=%.5f time=%.5f\n", test_loss, test_acc, timer_stop(TMR_TEST));
}
