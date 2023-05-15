#include "../include/gcn.cuh"

// #include "../include/rand.h"
// #include "../include/timer.h"

// ##################################################################################

void GCNParams::print_info() const
{
    std::cout << "num_nodes: " << num_nodes << std::endl;
    std::cout << "train_dim: " << train_dim << std::endl;
    std::cout << "val_dim: " << val_dim << std::endl;
    std::cout << "test_dim: " << test_dim << std::endl;
    std::cout << "input_dim: " << input_dim << std::endl;
    std::cout << "output_dim: " << output_dim << std::endl;
}

DevGCNData::DevGCNData(const GCNData &gcn_data) : dev_graph_index(DevSparseIndex(gcn_data.graph)), dev_feature_index(DevSparseIndex(gcn_data.feature_index))
{
    label_size = gcn_data.label.size();

    dev_feature_value = dev_shared_ptr<real>(dev_feature_index.indices_size);
    dev_graph_value = dev_shared_ptr<real>(dev_graph_index.indices_size);
    dev_split = dev_shared_ptr<natural>(label_size);
    dev_label = dev_shared_ptr<integer>(label_size);

    dev_feature_value.copy_to_device(gcn_data.feature_value.data());
    dev_graph_value.copy_to_device(gcn_data.graph_value.data());
    dev_split.copy_to_device(gcn_data.split.data());
    dev_label.copy_to_device(gcn_data.label.data());
}

// ##################################################################################

GCN::GCN(GCNParams *params_, GCNData *data_) : params(params_), data(data_), dev_data{DevGCNData(*data_)}
{
    initialize_random();
    initialize_truth();

    modules.reserve(8);
    variables.reserve(7);

    // dropout
    variables.push_back(std::make_shared<Variable>(data_->feature_index.indices.size(), false));
    input = variables.back();

    modules.push_back(std::make_unique<Dropout>(input, params->dropout, dev_rand_states));

    // sparse matmul
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->hidden_dim));
    shared_ptr<Variable> layer1_var1 = variables.back();

    variables.push_back(std::make_shared<Variable>(params->input_dim * params->hidden_dim, true, dev_rand_states));
    shared_ptr<Variable> layer1_weight = variables.back();
    layer1_weight->glorot(params->input_dim, params->hidden_dim);

    modules.push_back(std::make_unique<SparseMatmul>(input, layer1_weight, layer1_var1, &dev_data.dev_feature_index, params->num_nodes, params->input_dim, params->hidden_dim));

    // graph sum
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->hidden_dim));
    shared_ptr<Variable> layer1_var2 = variables.back();

    modules.push_back(std::make_unique<GraphSum>(layer1_var1, layer1_var2, &dev_data.dev_graph_index, dev_data.dev_graph_value, params->hidden_dim));

    // ReLU
    modules.push_back(std::make_unique<ReLU>(layer1_var2));

    // dropout
    modules.push_back(std::make_unique<Dropout>(layer1_var2, params->dropout, dev_rand_states));

    // dense matmul
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->output_dim));
    shared_ptr<Variable> layer2_var1 = variables.back();

    variables.push_back(std::make_shared<Variable>(params->hidden_dim * params->output_dim, true, dev_rand_states));
    shared_ptr<Variable> layer2_weight = variables.back();
    layer2_weight->glorot(params->hidden_dim, params->output_dim);

    modules.push_back(std::make_unique<Matmul>(layer1_var2, layer2_weight, layer2_var1, params->num_nodes, params->hidden_dim, params->output_dim));

    // graph sum
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->output_dim));
    output = variables.back();
    modules.push_back(std::make_unique<GraphSum>(layer2_var1, output, &dev_data.dev_graph_index, dev_data.dev_graph_value, params->output_dim));

    // cross entropy loss
    modules.push_back(std::make_unique<CrossEntropyLoss>(output, dev_truth, &loss, params->output_dim));

    // optimizer
    AdamParams adam_params = AdamParams::get_default();
    adam_params.lr = params->learning_rate;
    adam_params.weight_decay = params->weight_decay;

    std::vector<std::pair<shared_ptr<Variable>, bool>> vars = {{layer1_weight, true}, {layer2_weight, false}};
    optimizer = std::make_unique<Adam>(vars, adam_params);
}

// ##################################################################################

__global__ void initialize_random_kernel(randState *dev_rand_states)
{
    // curand_init(seed, index, offset, &state);
    curand_init(SEED, threadIdx.x, 0, &dev_rand_states[threadIdx.x]);
}

void GCN::initialize_random()
{
    dev_rand_states = dev_shared_ptr<randState>(N_THREADS);
    initialize_random_kernel<<<1, N_THREADS>>>(dev_rand_states.get());
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// ##################################################################################

void GCN::initialize_truth()
{
    dev_truth = dev_shared_ptr<integer>(params->num_nodes);
}

// ##################################################################################

__global__ void set_input_kernel(real *dev_data, real *dev_feature_value, natural size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dev_data[i] = dev_feature_value[i];
}

void GCN::set_input()
{
    dim3 n_blocks(CEIL(input->size, N_THREADS));
    dim3 n_threads(N_THREADS);
    set_input_kernel<<<n_blocks, n_threads>>>(input->dev_data.get(), dev_data.dev_feature_value.get(), input->size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ##################################################################################

__global__ void set_truth_kernel(integer *dev_truth, natural *dev_split, integer *dev_label, natural size, natural current_split)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dev_truth[i] = dev_split[i] == current_split ? dev_label[i] : -1;
}

void GCN::set_truth(int current_split)
{
    if (current_split == 1)
        modules.back()->set_num_samples(params->train_dim);
    else if (current_split == 2)
        modules.back()->set_num_samples(params->val_dim);
    else if (current_split == 3)
        modules.back()->set_num_samples(params->test_dim);

    dim3 n_blocks(CEIL(params->num_nodes, N_THREADS));
    dim3 n_threads(N_THREADS);
    set_truth_kernel<<<n_blocks, n_threads>>>(dev_truth.get(), dev_data.dev_split.get(), dev_data.dev_label.get(), params->num_nodes, current_split);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ##################################################################################

real GCN::get_l2_penalty()
{
    real l2 = 0;
    std::vector<real> weight(params->input_dim * params->hidden_dim);
    variables[2]->dev_data.copy_to_host(weight.data());
    for (const real &x : weight)
        l2 += x * x;
    // std::cout << "l2 loss: " << params->weight_decay * l2 / 2 << std::endl;
    return params->weight_decay * l2 / 2;
}

// ##################################################################################

std::pair<real, real> GCN::eval(natural current_split)
{
    set_input();
    set_truth(current_split);
    for (const auto &m : modules)
        m->forward(false);
    float test_loss = loss + get_l2_penalty();
    float test_acc = get_accuracy();
    return {test_loss, test_acc};
}

// ##################################################################################

real GCN::get_accuracy()
{
    std::vector<integer> truth(params->num_nodes);
    dev_truth.copy_to_host(truth.data());

    std::vector<real> output_data(params->num_nodes * params->output_dim);
    output->dev_data.copy_to_host(output_data.data());

    natural wrong = 0, total = 0;
    // iterate over training nodes
    for (int i = 0; i < params->num_nodes; i++)
    {
        if (truth[i] < 0)
            continue;
        total++;
        float truth_logit = output_data[i * params->output_dim + truth[i]];
        // more convenient: since I subtracted the max value of each row, simply
        // check if truth_logit < 0 instead of doing the following loop
        for (int j = 0; j < params->output_dim; j++)
            if (output_data[i * params->output_dim + j] > truth_logit)
            {
                wrong++;
                break;
            }
    }
    return float(total - wrong) / total;
}

// ##################################################################################

/**
 * Train an epoch of the model
 */
std::pair<real, real> GCN::train_epoch()
{
    // print_cpu(data->feature_value, data->feature_value.size());
    // print_gpu(dev_data.dev_feature_value, dev_data.dev_feature_index.indices_size, dev_data.dev_feature_index.indices_size);
    set_input();  // set the input data
    set_truth(1); // get the true labels for the dataset with split == 1 (train)

    // print_gpu<integer>(dev_truth, params->num_nodes, params->num_nodes);
    for (auto &m : modules) // iterate over the layer applying a forward pass
        m->forward(true);   // true means train
                            // correct the loss with the l2 regularization
    real train_loss = loss + get_l2_penalty();

    float train_acc = get_accuracy(); // compute the accuracy comparing the
                                      // prediction against the truth

    for (int i = modules.size() - 1; i >= 0; i--)
        modules[i]->backward(); // do a backward pass on the layers

    optimizer->step(); // apply a step of the adam optimization

    // return {train_loss, train_acc};

    // input->print(params->input_dim);
    //  print_gpu(dev_data.dev_feature_value, dev_data.dev_feature_index.indices_size, dev_data.dev_feature_index.indices_size);
    /*
        std::cout << "layer1_var1" << std::endl;
        variables[1]->print(params->hidden_dim);
        std::cout << "layer1_weight" << std::endl;
        variables[2]->print(params->hidden_dim);
        std::cout << "layer1_var2" << std::endl;
        variables[3]->print(params->hidden_dim);
        std::cout << "layer2_var1" << std::endl;
        variables[4]->print(params->output_dim);
        std::cout << "layer2_weight" << std::endl;
        variables[5]->print(params->output_dim);
        std::cout << "output" << std::endl;
        variables[6]->print(params->output_dim);
    */
    return {train_loss, train_acc};
}

// ##################################################################################

void GCN::run()
{
    natural epoch = 1;
    real total_time = 0.0;
    std::vector<real> loss_history;
    // Iterate the training process based on the selected number of epoch
    for (; epoch <= params->epochs; epoch++)
    {
        real train_loss, train_acc, val_loss, val_acc;
        timer_start(TMR_TRAIN); // just for timing purposes
        std::tie(train_loss, train_acc) =
            train_epoch(); // train the epoch and record the current train_loss and
                           // train_accuracy
                           // eval the model at the current step in order to obtain the val_loss and val_accuracy

        std::tie(val_loss, val_acc) = eval(2);
        /*
        printf("epoch=%d train_loss=%.5f"
               "time=%.5f\n",
               epoch, train_loss, timer_stop(TMR_TRAIN));
    */
        printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f "
               "time=%.5f\n",
               epoch, train_loss, train_acc, val_loss, val_acc,
               timer_stop(TMR_TRAIN));
        /*
        loss_history.push_back(val_loss); // record the validation loss in order to
                                  // apply an early stopping mechanism

        // early stopping mechanism
        if (params.early_stopping > 0 && epoch >= params.early_stopping) {
        float recent_loss = 0.0;
        for (int i = epoch - params.early_stopping; i < epoch; i++)
        recent_loss += loss_history[i];
        if (val_loss > recent_loss / params.early_stopping) {
        printf("Early stopping...\n");
        break;
        }

        }
        */
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
    printf("test_loss=%.5f test_acc=%.5f time=%.5f\n", test_loss, test_acc,
           timer_stop(TMR_TEST));
}

// ##################################################################################