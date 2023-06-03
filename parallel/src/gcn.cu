#include "../include/gcn.cuh"

// ##################################################################################

GCNSmartObjects::GCNSmartObjects(const natural n_layers) : forward_training_stream(High), forward_evaluation_stream(High)
{
    backward_streams.emplace_back(High);
    backward_streams.emplace_back(High);
    start_matmul_forward.resize(n_layers);
    start_matmul_backward.resize(n_layers - 1);
}

// ##################################################################################

void GCNParams::print_info() const
{
    std::cout << std::endl;
    std::cout << "PARSED PARAMETERS FROM DATA:" << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
    std::cout << "Number of features: " << input_dim << std::endl;
    std::cout << "Number of labels: " << output_dim << std::endl;
    std::cout << "Training dataset dimension: " << train_dim << std::endl;
    std::cout << "Validation dataset dimension: " << val_dim << std::endl;
    std::cout << "Test dataset dimension: " << test_dim << std::endl;
    std::cout << std::endl;
}

// ##################################################################################

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

void GCN::insert_first_layer()
{
    // dropout
    variables.push_back(std::make_shared<Variable>(data->feature_index.indices.size(), false));
    input = variables.back();
    set_input(smart_objects.forward_training_stream, true);
    modules.push_back(std::make_unique<Dropout>(input, params->dropouts.front(), dev_rand_states));

    // sparse matmul
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->hidden_dims.front()));
    shared_ptr<Variable> layer1_var1 = variables.back();

    variables.push_back(std::make_shared<Variable>(params->input_dim * params->hidden_dims.front(), true, true));
    shared_ptr<Variable> layer1_weight = variables.back();
    layer1_weight->glorot(params->input_dim, params->hidden_dims.front());
    // layer1_weight->set_value(0.5, smart_objects.forward_training_stream);
    weights.push_back(layer1_weight);
    dev_l2 = dev_shared_ptr<real>(1); // used by get_l2_penalty()
    pinned_l2 = pinned_host_ptr<real>(1);

    modules.push_back(std::make_unique<SparseMatmul>(input, layer1_weight, layer1_var1, &dev_data.dev_feature_index, params->num_nodes, params->input_dim, params->hidden_dims.front(), smart_objects.start_matmul_forward.front(), smart_objects.start_set_input));

    // graph sum
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->hidden_dims.front()));
    shared_ptr<Variable> layer1_var2 = variables.back();

    smart_event trash;
    modules.push_back(std::make_unique<GraphSum>(layer1_var1, layer1_var2, &dev_data.dev_graph_index, dev_data.dev_graph_value, params->hidden_dims.front(), false, trash));

    // ReLU
    modules.push_back(std::make_unique<ReLU>(layer1_var2));
}

// ##################################################################################

void GCN::insert_layer(const natural input_dim, const natural output_dim, const real dropout, const natural layer_index)
{
    // layer_index starts from 1
    // dropout
    shared_ptr<Variable> layer_prev_var2 = variables.back();
    modules.push_back(std::make_unique<Dropout>(layer_prev_var2, dropout, dev_rand_states));

    // dense matmul
    variables.push_back(std::make_shared<Variable>(params->num_nodes * output_dim));
    shared_ptr<Variable> layer_var1 = variables.back();

    variables.push_back(std::make_shared<Variable>(params->input_dim * params->output_dim, true, true));
    shared_ptr<Variable> layer_weight = variables.back();
    layer_weight->glorot(input_dim, output_dim);
    // layer_weight->set_value(0.5, forward_training_stream);
    weights.push_back(layer_weight);

    modules.push_back(std::make_unique<Matmul>(layer_prev_var2, layer_weight, layer_var1, params->num_nodes, input_dim, output_dim, smart_objects.start_matmul_forward[layer_index], smart_objects.start_matmul_backward[L - layer_index - 1], smart_objects.backward_streams[1]));

    // graph sum
    variables.push_back(std::make_shared<Variable>(params->num_nodes * output_dim));
    shared_ptr<Variable> layer_var2 = variables.back();
    modules.push_back(std::make_unique<GraphSum>(layer_var1, layer_var2, &dev_data.dev_graph_index, dev_data.dev_graph_value, output_dim, true, smart_objects.start_matmul_backward[L - layer_index - 1]));

    // ReLU
    modules.push_back(std::make_unique<ReLU>(layer_var2));
}

// ##################################################################################

void GCN::insert_last_layer()
{
    // dropout
    shared_ptr<Variable> layer_prev_var2 = variables.back();
    modules.push_back(std::make_unique<Dropout>(layer_prev_var2, params->dropouts.back(), dev_rand_states));

    // dense matmul
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->output_dim));
    shared_ptr<Variable> layer_last_var1 = variables.back();

    variables.push_back(std::make_shared<Variable>(params->hidden_dims.back() * params->output_dim, true, true));
    shared_ptr<Variable> layer_last_weight = variables.back();
    layer_last_weight->glorot(params->hidden_dims.back(), params->output_dim);
    // layer_last_weight->set_value(0.5, smart_objects.forward_training_stream);
    weights.push_back(layer_last_weight);

    modules.push_back(std::make_unique<Matmul>(layer_prev_var2, layer_last_weight, layer_last_var1, params->num_nodes, params->hidden_dims.back(), params->output_dim, smart_objects.start_matmul_forward.back(), smart_objects.start_matmul_backward.front(), smart_objects.backward_streams[1]));

    // graph sum
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->output_dim));
    output = variables.back();
    modules.push_back(std::make_unique<GraphSum>(layer_last_var1, output, &dev_data.dev_graph_index, dev_data.dev_graph_value, params->output_dim, true, smart_objects.start_matmul_backward.front()));

    // cross entropy loss
    modules.push_back(std::make_unique<CrossEntropyLoss>(output, dev_truth, loss, params->output_dim, smart_objects.start_backward));
}

// ##################################################################################

GCN::GCN(GCNParams const *params_, AdamParams const *adam_params_, GCNData const *data_) : params(params_), adam_params(adam_params_), data(data_), dev_data{DevGCNData(*data_)}, smart_objects(params_->n_layers)
{
    L = params->n_layers;
    initialize_random();
    dev_truth = dev_shared_ptr<integer>(params->num_nodes);
    dev_wrong = dev_shared_ptr<natural>(1); // used by get_accuracy()
    pinned_wrong = pinned_host_ptr<natural>(1);
    loss = pinned_host_ptr<real>(1);

    modules.reserve(4 * L);
    variables.reserve(4 + 3 * (L - 1));
    weights.reserve(L);
    decays.resize(L, false); // true for L2 regularization
    decays.front() = true;

    insert_first_layer();

    // add layer 1 to n_layers - 2
    for (natural i = 1; i < L - 1; ++i)
        insert_layer(params->hidden_dims[i - 1], params->hidden_dims[i], params->dropouts[i], i);

    insert_last_layer();

    // optimizer
    optimizer = Adam(weights, decays, adam_params, smart_objects.backward_streams, smart_objects.start_matmul_forward, smart_objects.forward_training_stream);
}

// ##################################################################################

__global__ void initialize_random_kernel(randState *dev_rand_states, const natural seed)
{
    // curand_init(seed, index, offset, &state);
    curand_init(seed * threadIdx.x, threadIdx.x, 0, &dev_rand_states[threadIdx.x]);
}

void GCN::initialize_random()
{
    dev_rand_states = dev_shared_ptr<randState>(CudaParams::N_THREADS_DROPOUT);
    initialize_random_kernel<<<1, CudaParams::N_THREADS_DROPOUT, 0, smart_objects.forward_training_stream.get()>>>(dev_rand_states.get(), CudaParams::SEED);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    cudaStreamSynchronize(smart_objects.forward_training_stream.get());
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

void GCN::set_input(smart_stream stream, bool first) const
{
    const natural n_blocks = std::min(CEIL(input->size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    if (!first)
        cudaStreamWaitEvent(stream.get(), smart_objects.start_set_input.get());
    set_input_kernel<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(input->dev_data.get(), dev_data.dev_feature_value.get(), input->size);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// ##################################################################################

__global__ void set_truth_kernel(integer *dev_truth, const natural *dev_split, const integer *dev_label, const natural size, const natural current_split)
{
    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < size; i += blockDim.x * gridDim.x)
        dev_truth[i] = dev_split[i] == current_split ? dev_label[i] : -1;
}

void GCN::set_truth(const natural current_split, smart_stream stream) const
{
    if (current_split == 1)
        modules.back()->set_num_samples(params->train_dim);
    else if (current_split == 2)
        modules.back()->set_num_samples(params->val_dim);
    else if (current_split == 3)
        modules.back()->set_num_samples(params->test_dim);

    const natural n_blocks = std::min(CEIL(params->num_nodes, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    set_truth_kernel<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(dev_truth.get(), dev_data.dev_split.get(), dev_data.dev_label.get(), params->num_nodes, current_split);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
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

void GCN::get_l2_penalty(smart_stream stream) const
{
    dev_l2.set_zero(stream);
    natural i = 0;
    for (const auto &weight : weights)
        if (decays[i++])
        {
            const natural n_blocks = std::min(CEIL(weight->size, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
            get_l2_penalty_kernel<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(dev_l2.get(), weight->dev_data.get(), weight->size);
#ifdef DEBUG_CUDA
            CHECK_CUDA_ERROR(cudaGetLastError());
#endif
        }

    dev_l2.copy_to_host_async(pinned_l2.get(), stream);
}

// ##################################################################################

__global__ void get_accuracy_kernel(const integer *dev_truth, const real *dev_data, natural *wrong, const natural num_nodes, const natural output_dim)
{

    natural id = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (natural i = id; i < num_nodes; i += blockDim.x * gridDim.x)
    {
        if (dev_truth[i] < 0)
            continue;
        const real truth_logit = dev_data[i * output_dim + dev_truth[i]];

        if (truth_logit < 0)
            atomicAdd(wrong, 1);
    }
}

void GCN::get_accuracy(smart_stream stream) const
{
    dev_wrong.set_zero(stream);
    const natural n_blocks = std::min(CEIL(params->num_nodes, CudaParams::N_THREADS), CudaParams::N_BLOCKS);
    get_accuracy_kernel<<<n_blocks, CudaParams::N_THREADS, 0, stream.get()>>>(dev_truth.get(), output->dev_data.get(), dev_wrong.get(), params->num_nodes, params->output_dim);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
    dev_wrong.copy_to_host_async(pinned_wrong.get(), stream);
}

// ##################################################################################

std::pair<real, real> GCN::eval(const natural current_split) const
{
    set_input(smart_objects.forward_evaluation_stream, false);
    set_truth(current_split, smart_objects.forward_evaluation_stream);
    for (const auto &m : modules)
        m->forward(false, smart_objects.forward_evaluation_stream);

    get_l2_penalty(smart_objects.forward_evaluation_stream);
    get_accuracy(smart_objects.forward_evaluation_stream);
    return finalize(smart_objects.forward_evaluation_stream);
}

// ##################################################################################

std::pair<real, real> GCN::train_epoch()
{
    // input set by constructor and by eval()
    set_truth(1, smart_objects.forward_training_stream);         // get the true labels for the dataset with split == 1 (train)
    for (const auto &m : modules)                                // iterate over the layer applying a forward pass
        m->forward(true, smart_objects.forward_training_stream); // true means train

    get_l2_penalty(smart_objects.forward_training_stream);
    get_accuracy(smart_objects.forward_training_stream);

    for (integer i = modules.size() - 1; i >= 0; i--)
        modules[i]->backward(smart_objects.backward_streams[0]); // do a backward pass on the layers

    optimizer.step(); // apply a step of the adam optimization

    // DEBUG
    /*
        std::cout << "layer1_var1" << std::endl;
        variables[1]->print("data", params->hidden_dim);
        std::cout << "layer1_weight" << std::endl;
        variables[2]->print("data", params->hidden_dim);
        std::cout << "layer1_var2" << std::endl;
        variables[3]->print("data", params->hidden_dim);
        std::cout << "layer2_var1" << std::endl;
        variables[4]->print("data", params->output_dim);
        std::cout << "layer2_weight" << std::endl;
        variables[5]->print("data", params->output_dim);
        std::cout << "output" << std::endl;
        variables[6]->print("data", params->output_dim);
    */
    /*
     cudaDeviceSynchronize();
     for (unsigned int i = 0; i < variables.size(); i++)
         variables[i]->save("variable" + std::to_string(i) + ".txt", "data", params->output_dim);
 */
    return finalize(smart_objects.forward_training_stream);
}

// ##################################################################################

void GCN::run()
{
    timer_start(TMR_TOTAL);
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

        real val_loss{0.f}, val_acc{0.f};
        // eval the model at the current step in order to obtain the val_loss and val_accuracy
        std::tie(val_loss, val_acc) = eval(2);
#ifndef TUNE_CUDA
        const auto time = timer_stop(TMR_TRAIN);
        printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f time=%.5f\n",
               epoch, train_loss, train_acc, val_loss, val_acc, time);
#else
        timer_stop(TMR_TRAIN);
#endif

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
    }
    timer_stop(TMR_TOTAL);
#ifndef TUNE_CUDA
    PRINT_TIMER_AVERAGE(TMR_TRAIN, epoch);
#else
    PRINT_TIMER_AVERAGE_TUNE_CUDA(TMR_TRAIN, epoch);
#endif

    /*
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
    */
#ifndef TUNE_CUDA
    real test_loss, test_acc;
    timer_start(TMR_TEST);
    std::tie(test_loss, test_acc) = eval(3); // eval the model on the test set

    printf("test_loss=%.5f test_acc=%.5f time=%.5f\n", test_loss, test_acc, timer_stop(TMR_TEST));
    printf("total time: %.5f\n", timer_total(TMR_TOTAL));
#endif
}

// ##################################################################################

std::pair<real, real> GCN::finalize(smart_stream stream) const
{
    // syncronize the stream
    cudaStreamSynchronize(stream.get());
    const natural total = modules.back()->get_num_samples();

    // loss
    *loss /= total;

    real l2 = adam_params->weight_decay * (*pinned_l2) / real(2);
    const real final_loss = *loss + l2;

    // accuracy
    const real final_accuracy = static_cast<real>(total - *pinned_wrong) / static_cast<real>(total);
    return {final_loss, final_accuracy};
}