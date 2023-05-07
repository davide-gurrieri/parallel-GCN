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

// DevGCNData::DevGCNData() : dev_graph(DevSparseIndex()), dev_feature_index(DevSparseIndex()), dev_feature_value(nullptr), dev_split(nullptr), dev_label(nullptr), label_size(0) {}

DevGCNData::DevGCNData(const GCNData &gcn_data) : dev_graph(DevSparseIndex(gcn_data.graph)), dev_feature_index(DevSparseIndex(gcn_data.feature_index))
{
    label_size = gcn_data.label.size();

    CHECK_CUDA_ERROR(cudaMalloc(&dev_feature_value, dev_feature_index.indices_size * sizeof(real)));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_split, label_size * sizeof(natural)));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_label, label_size * sizeof(natural)));

    CHECK_CUDA_ERROR(cudaMemcpy(dev_feature_value, gcn_data.feature_value.data(), dev_feature_index.indices_size * sizeof(real), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_split, gcn_data.split.data(), label_size * sizeof(natural), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_label, gcn_data.label.data(), label_size * sizeof(natural), cudaMemcpyHostToDevice));
}

DevGCNData::~DevGCNData()
{
    if (dev_feature_value)
        CHECK_CUDA_ERROR(cudaFree(dev_feature_value));
    if (dev_split)
        CHECK_CUDA_ERROR(cudaFree(dev_split));
    if (dev_label)
        CHECK_CUDA_ERROR(cudaFree(dev_label));
}

GCN::GCN(GCNParams *params_, GCNData *data_) : params(params_), data(data_), dev_data{DevGCNData(*data_)}
{
    count++;
    initialize_random();

    /*
        CHECK_CUDA_ERROR(cudaMalloc(&params_dev, sizeof(GCNParams)));
        CHECK_CUDA_ERROR(cudaMemcpy(params_dev, params_, sizeof(GCNParams), cudaMemcpyHostToDevice));
    */

    modules.reserve(8);
    variables.reserve(7);

    // dropout
    variables.push_back(new Variable(data_->feature_index.indices.size(), false));
    input = variables.back();
    modules.push_back(new Dropout(input, params->dropout, dev_rand_states));

    // sparse matmul
    variables.push_back(new Variable(params->num_nodes * params->hidden_dim));
    Variable *layer1_var1 = variables.back();

    variables.push_back(new Variable(params->input_dim * params->hidden_dim, true, dev_rand_states));
    Variable *layer1_weight = variables.back();
    layer1_var1->glorot(params->input_dim, params->hidden_dim);

    modules.push_back(new SparseMatmul(input, layer1_weight, layer1_var1, &dev_data.dev_feature_index, params->num_nodes, params->input_dim, params->hidden_dim));

    // graph sum
    variables.push_back(new Variable(params->num_nodes * params->hidden_dim));
    Variable *layer1_var2 = variables.back();

    modules.push_back(new GraphSum(layer1_var1, layer1_var2, &dev_data.dev_graph, params->hidden_dim));

    // ReLU
    modules.push_back(new ReLU(layer1_var2));

    // dropout
    modules.push_back(new Dropout(layer1_var2, params->dropout, dev_rand_states));

    // dense matmul
    variables.push_back(new Variable(params->num_nodes * params->output_dim));
    Variable *layer2_var1 = variables.back();

    variables.push_back(new Variable(params->hidden_dim * params->output_dim, true, dev_rand_states));
    Variable *layer2_weight = variables.back();
    layer2_weight->glorot(params->hidden_dim, params->output_dim);

    modules.push_back(new Matmul(layer1_var2, layer2_weight, layer2_var1, params->num_nodes, params->hidden_dim, params->output_dim));

    // graph sum
    variables.push_back(new Variable(params->num_nodes * params->output_dim));
    output = variables.back();
    modules.push_back(new GraphSum(layer2_var1, output, &dev_data.dev_graph, params->output_dim));

    // cross entropy loss
    modules.push_back(new CrossEntropyLoss(output, dev_data.dev_label, &loss, params->output_dim));

    /*

         // optimizer
         AdamParams adam_params = AdamParams::get_default();
         adam_params->lr = params->learning_rate;
         adam_params->weight_decay = params->weight_decay;
         optimizer = new Adam({{layer1_weight, true}, {layer2_weight, false}}, adam_params);
         */
}

GCN::~GCN()
{
    std::cout << count << std::endl;
    CHECK_CUDA_ERROR(cudaFree(dev_rand_states));
}

__global__ void initialize_random_kernel(curandState *dev_rand_states)
{
    // curand_init(seed, index, offset, &state);
    curand_init(SEED + threadIdx.x, threadIdx.x, 0, &dev_rand_states[threadIdx.x]);
}

/// @brief initializes an array of curandState structures in device memory
void GCN::initialize_random()
{
    CHECK_CUDA_ERROR(cudaMalloc(&dev_rand_states, N_THREADS * sizeof(curandState)));
    initialize_random_kernel<<<1, N_THREADS>>>(dev_rand_states);
    CHECK_CUDA_ERROR(cudaGetLastError());
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
