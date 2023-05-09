#include "../include/gcn.cuh"

// #include "../include/rand.h"
// #include "../include/timer.h"

/**
 * Returns the default paramets of the model
 * they will be overwritten by the parser when reading the dataset
 */
GCNParams GCNParams::get_default()
{
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
    return {      // CORA
            2708, // num_nodes
            1433, // input_dim
            16,   // hidden_dim
            7,    // output_dim
            0.5,  // dropouyt
            0.01, // learning_rate
            5e-4, // weight_decay
            100,  // epochs
            0};   // early_stopping
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

// ##################################################################################

DevGCNData::DevGCNData(const GCNData &gcn_data) : dev_graph(DevSparseIndex(gcn_data.graph)), dev_feature_index(DevSparseIndex(gcn_data.feature_index))
{
    label_size = gcn_data.label.size();

    dev_feature_value = dev_shared_ptr<real>(dev_feature_index.indices_size);
    dev_split = dev_shared_ptr<natural>(label_size);
    dev_label = dev_shared_ptr<integer>(label_size);

    dev_feature_value.copy_to_device(gcn_data.feature_value.data());
    dev_split.copy_to_device(gcn_data.split.data());
    dev_label.copy_to_device(gcn_data.label.data());
}

// ##################################################################################

GCN::GCN(GCNParams *params_, GCNData *data_) : params(params_), data(data_), dev_data{DevGCNData(*data_)}
{
    initialize_random();
    initialize_truth();

    /*
        CHECK_CUDA_ERROR(cudaMalloc(&params_dev, sizeof(GCNParams)));
        CHECK_CUDA_ERROR(cudaMemcpy(params_dev, params_, sizeof(GCNParams), cudaMemcpyHostToDevice));
    */

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

    modules.push_back(std::make_unique<GraphSum>(layer1_var1, layer1_var2, &dev_data.dev_graph, params->hidden_dim));

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
    modules.push_back(std::make_unique<GraphSum>(layer2_var1, output, &dev_data.dev_graph, params->output_dim));

    // cross entropy loss
    modules.push_back(std::make_unique<CrossEntropyLoss>(output, dev_data.dev_label, &loss, params->output_dim));

    // optimizer
    AdamParams adam_params = AdamParams::get_default();
    adam_params.lr = params->learning_rate;
    adam_params.weight_decay = params->weight_decay;

    std::vector<std::pair<shared_ptr<Variable>, bool>> vars = {{layer1_weight, true}, {layer2_weight, false}};
    optimizer = std::make_unique<Adam>(vars, adam_params);
}

// ##################################################################################

__global__ void initialize_random_kernel(curandState *dev_rand_states)
{
    // curand_init(seed, index, offset, &state);
    curand_init(SEED + threadIdx.x, threadIdx.x, 0, &dev_rand_states[threadIdx.x]);
}

void GCN::initialize_random()
{
    dev_rand_states = dev_shared_ptr<curandState>(N_THREADS);
    initialize_random_kernel<<<1, N_THREADS>>>(dev_rand_states.get());
    CHECK_CUDA_ERROR(cudaGetLastError());
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
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
    dim3 n_blocks(CEIL(params->num_nodes, N_THREADS));
    dim3 n_threads(N_THREADS);
    set_truth_kernel<<<n_blocks, n_threads>>>(dev_truth.get(), dev_data.dev_split.get(), dev_data.dev_label.get(), params->num_nodes, current_split);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ##################################################################################