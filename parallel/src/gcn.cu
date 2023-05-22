#include "../include/gcn.cuh"

// ##################################################################################

void GCNParams::print_info() const
{
    std::cout << std::endl;
    std::cout << "PARSED PARAMETERS:" << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "Number of features: " << input_dim << std::endl;
    std::cout << "Number of labels: " << output_dim << std::endl;
    std::cout << "Training dataset dimension: " << train_dim << std::endl;
    std::cout << "Validation dataset dimension: " << val_dim << std::endl;
    std::cout << "Test dataset dimension: " << test_dim << std::endl;
    std::cout << std::endl;
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

GCN::GCN(GCNParams const *params_, AdamParams const *adam_params_, GCNData const *data_) : params(params_), adam_params(adam_params_), data(data_), dev_data{DevGCNData(*data_)}
{
    streams.emplace_back(High); // forward + l2_loss + get_accuracy
    streams.emplace_back(High); // backward + optimization 1
    streams.emplace_back(High); // backard + optimization 2
    events.resize(4);
    initialize_random();
    dev_truth = dev_shared_ptr<integer>(params->num_nodes);
    dev_wrong = dev_shared_ptr<natural>(1); // used by get_accuracy()
    pinned_wrong = pinned_host_ptr<natural>(1);
    loss = pinned_host_ptr<real>(1);

    modules.reserve(8);
    variables.reserve(7);

    // dropout
    variables.push_back(std::make_shared<Variable>(data_->feature_index.indices.size(), false));
    input = variables.back();
    set_input();

    modules.push_back(std::make_unique<Dropout>(input, params->dropout, dev_rand_states));

    // sparse matmul
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->hidden_dim));
    shared_ptr<Variable> layer1_var1 = variables.back();

    variables.push_back(std::make_shared<Variable>(params->input_dim * params->hidden_dim, true, dev_rand_states));
    shared_ptr<Variable> layer1_weight = variables.back();
    layer1_weight->glorot(params->input_dim, params->hidden_dim);
    dev_l2 = dev_shared_ptr<real>(1); // used by get_l2_penalty()
    pinned_l2 = pinned_host_ptr<real>(1);

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
    modules.push_back(std::make_unique<CrossEntropyLoss>(output, dev_truth, loss, params->output_dim));

    // optimizer
    optimizer = Adam({{layer1_weight, true}, {layer2_weight, false}}, adam_params);
}

// ##################################################################################

__global__ void initialize_random_kernel(randState *dev_rand_states)
{
    // curand_init(seed, index, offset, &state);
    curand_init(SEED, 0, threadIdx.x, &dev_rand_states[threadIdx.x]);
}

void GCN::initialize_random()
{
    dev_rand_states = dev_shared_ptr<randState>(N_THREADS);
    initialize_random_kernel<<<1, N_THREADS, 0, streams[0].get()>>>(dev_rand_states.get());
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    cudaStreamSynchronize(streams[0].get());
}

// ##################################################################################

__global__ void set_input_kernel(real *dev_data, const real *dev_feature_value, const natural size)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
    {
        dev_data[i] = dev_feature_value[i];
    }
}

void GCN::set_input() const
{
    const natural n_blocks = std::min(CEIL(input->size, N_THREADS), N_BLOCKS);
    cudaStreamWaitEvent(streams[0].get(), events[2].get());
    set_input_kernel<<<n_blocks, N_THREADS, 0, streams[0].get()>>>(input->dev_data.get(), dev_data.dev_feature_value.get(), input->size);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    // cudaStreamSynchronize(streams[0].get());
}

// ##################################################################################

__global__ void set_truth_kernel(integer *dev_truth, const natural *dev_split, const integer *dev_label, const natural size, const natural current_split)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
        dev_truth[i] = dev_split[i] == current_split ? dev_label[i] : -1;
}

void GCN::set_truth(const natural current_split) const
{
    if (current_split == 1)
        modules.back()->set_num_samples(params->train_dim);
    else if (current_split == 2)
        modules.back()->set_num_samples(params->val_dim);
    else if (current_split == 3)
        modules.back()->set_num_samples(params->test_dim);

    const natural n_blocks = std::min(CEIL(params->num_nodes, N_THREADS), N_BLOCKS);
    set_truth_kernel<<<n_blocks, N_THREADS, 0, streams[0].get()>>>(dev_truth.get(), dev_data.dev_split.get(), dev_data.dev_label.get(), params->num_nodes, current_split);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    // cudaStreamSynchronize(streams[0].get());
}

// ##################################################################################

__global__ void get_l2_penalty_kernel(real *dev_l2, const real *weight, const natural size)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
    const natural warp_size = 32;
    real sum = static_cast<real>(0); // Initialize partial sum for this thread;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
    {
        sum += weight[i] * weight[i];
    }
    sum = warp_reduce(sum);
    if ((threadIdx.x & (warp_size - 1)) == 0)
        atomicAdd(dev_l2, sum);
}

real GCN::get_l2_penalty() const
{
    dev_l2.set_zero(streams[0]);
    const auto &weights = variables[2];
    const natural n_blocks = std::min(CEIL(weights->size, N_THREADS), N_BLOCKS);
    get_l2_penalty_kernel<<<n_blocks, N_THREADS, 0, streams[0].get()>>>(dev_l2.get(), weights->dev_data.get(), weights->size);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    // cudaStreamSynchronize(streams[0].get());
    dev_l2.copy_to_host_async(pinned_l2.get(), streams[0]);
    cudaStreamSynchronize(streams[0].get());
    return adam_params->weight_decay * (*pinned_l2) / real(2);
}

// ##################################################################################

std::pair<real, real> GCN::eval(const natural current_split) const
{
    set_input();
    set_truth(current_split);
    for (const auto &m : modules)
        m->forward(false);
    cudaStreamSynchronize(streams[0].get());
    *loss /= modules.back()->get_num_samples();
    const real test_loss = *loss + get_l2_penalty();
    const real test_acc = get_accuracy();
    return {test_loss, test_acc};
}

// ##################################################################################

__global__ void get_accuracy_kernel(const integer *dev_truth, const real *dev_data, natural *wrong, const natural num_nodes, const natural output_dim)
{

    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < num_nodes; i += blockDim.x * gridDim.x)
    {
        if (dev_truth[i] < 0)
            return;
        const real truth_logit = dev_data[i * output_dim + dev_truth[i]];
        if (truth_logit < static_cast<real>(0))
            atomicAdd(wrong, 1);
    }
}

real GCN::get_accuracy() const
{
    dev_wrong.set_zero(streams[0]);
    const natural n_blocks = std::min(CEIL(params->num_nodes, N_THREADS), N_BLOCKS);
    get_accuracy_kernel<<<n_blocks, N_THREADS, 0, streams[0].get()>>>(dev_truth.get(), output->dev_data.get(), dev_wrong.get(), params->num_nodes, params->output_dim);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    // cudaStreamSynchronize(streams[0].get());
    dev_wrong.copy_to_host_async(pinned_wrong.get(), streams[0]);
    cudaStreamSynchronize(streams[0].get());
    const natural total = modules.back()->get_num_samples();
    return static_cast<real>(total - *pinned_wrong) / static_cast<real>(total);
}

// ##################################################################################

std::pair<real, real> GCN::train_epoch()
{
// input set by constructor and by eval()
#ifndef VERBOSE
    set_input();
#endif
    set_truth(1); // get the true labels for the dataset with split == 1 (train)

    for (const auto &m : modules) // iterate over the layer applying a forward pass
        m->forward(true);         // true means train

    for (integer i = modules.size() - 1; i >= 0; i--)
        modules[i]->backward(); // do a backward pass on the layers

    optimizer.step(); // apply a step of the adam optimization

    // DEBUG
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

#ifdef VERBOSE
    cudaStreamSynchronize(streams[0].get());
    *loss /= modules.back()->get_num_samples();
    // correct the loss with the l2 regularization
    const real train_loss = *loss + get_l2_penalty();
    // compute the accuracy comparing the prediction against the truth
    const real train_acc = get_accuracy();
    return {train_loss, train_acc};
#else
    return {0., 0.};
#endif
}

// ##################################################################################

void GCN::run()
{
    natural epoch = 1;
    // real total_time = 0.0;
    std::vector<real> loss_history;
    loss_history.reserve(params->epochs);
    // Iterate the training process based on the selected number of epoch
    for (; epoch <= params->epochs; epoch++)
    {
        real train_loss{0.f}, train_acc{0.f};

        timer_start(TMR_TRAIN); // just for timing purposes
        // train the epoch and record the current train_loss and train_accuracy
        std::tie(train_loss, train_acc) = train_epoch();
#ifdef VERBOSE
        real val_loss{0.f}, val_acc{0.f};
        // eval the model at the current step in order to obtain the val_loss and val_accuracy
        std::tie(val_loss, val_acc) = eval(2);

        printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f time=%.5f\n",
               epoch, train_loss, train_acc, val_loss, val_acc, timer_stop(TMR_TRAIN));

        if (params->early_stopping > 0)
        {
            // record the validation loss in order to apply an early stopping mechanism
            loss_history.push_back(val_loss);
            // early stopping mechanism
            if (epoch >= params->early_stopping)
            {
                real recent_loss = 0.0;
                for (natural i = epoch - params->early_stopping; i < epoch; i++)
                    recent_loss += loss_history[i];
                if (val_loss > recent_loss / static_cast<real>(params->early_stopping))
                {
                    printf("Early stopping...\n");
                    break;
                }
            }
        }
#else
        timer_stop(TMR_TRAIN);
#endif
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
    PRINT_TIMER_AVERAGE(TMR_OPTIMIZER, epoch);

    real test_loss, test_acc;
    timer_start(TMR_TEST);
    std::tie(test_loss, test_acc) = eval(3); // eval the model on the test set
    printf("test_loss=%.5f test_acc=%.5f time=%.5f\n", test_loss, test_acc, timer_stop(TMR_TEST));
}

// ##################################################################################