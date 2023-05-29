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

#ifdef FEATURE
    dev_feature_value = dev_shared_ptr<real>(dev_feature_index.indices_size);
#endif
    dev_graph_value = dev_shared_ptr<real>(dev_graph_index.indices_size);
    dev_split = dev_shared_ptr<natural>(label_size);
    dev_label = dev_shared_ptr<integer>(label_size);

#ifdef FEATURE
    dev_feature_value.copy_to_device(gcn_data.feature_value.data());
#endif
    dev_graph_value.copy_to_device(gcn_data.graph_value.data());
    dev_split.copy_to_device(gcn_data.split.data());
    dev_label.copy_to_device(gcn_data.label.data());
}

// ##################################################################################

GCN::GCN(GCNParams const *params_, AdamParams const *adam_params_, GCNData const *data_) : params(params_), adam_params(adam_params_), data(data_), dev_data{DevGCNData(*data_)}
{
    streams.emplace_back(High); // forward training + l2_loss + get_accuracy
    streams.emplace_back(High); // backward + optimization 1
    streams.emplace_back(High); // backard + optimization 2
    streams.emplace_back(High); // forward evaluation
    events.resize(8 + 5 + 2);
    initialize_random();
    dev_truth = dev_shared_ptr<integer>(params->num_nodes);

    dev_wrong_train = dev_shared_ptr<real>(1); // used by get_accuracy()
    dev_l2_train = dev_shared_ptr<real>(1);    // used by get_l2_penalty()
    dev_loss_train = dev_shared_ptr<real>(1);

    dev_wrong_eval = dev_shared_ptr<real>(1); // used by get_accuracy()
    dev_l2_eval = dev_shared_ptr<real>(1);    // used by get_l2_penalty()
    dev_loss_eval = dev_shared_ptr<real>(1);

    dev_loss_history = dev_shared_ptr<real>(params->epochs);
    dev_interrupt = dev_shared_ptr<natural>(1);
    dev_interrupt.set_zero(streams[0]);

    modules.reserve(8);
    variables.reserve(7);

#ifdef FEATURE
    // dropout
    variables.push_back(std::make_shared<Variable>(data_->feature_index.indices.size(), false));
    input = variables.back();
    set_input(streams[0], true);
    modules.push_back(std::make_unique<Dropout>(input, params->dropout_input, dev_rand_states));
#else
    // for compatibility
    variables.push_back(std::make_shared<Variable>());
    input = variables.back();
#endif

    // sparse matmul
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->hidden_dim));
    shared_ptr<Variable> layer1_var1 = variables.back();

    variables.push_back(std::make_shared<Variable>(params->input_dim * params->hidden_dim, true, dev_rand_states));
    shared_ptr<Variable> layer1_weight = variables.back();
    layer1_weight->glorot(params->input_dim, params->hidden_dim);
    // layer1_weight->set_value(0.5, streams[0]);

    modules.push_back(std::make_unique<SparseMatmul>(input, layer1_weight, layer1_var1, &dev_data.dev_feature_index, params->num_nodes, params->input_dim, params->hidden_dim));

    // graph sum
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->hidden_dim));
    shared_ptr<Variable> layer1_var2 = variables.back();

    modules.push_back(std::make_unique<GraphSum>(layer1_var1, layer1_var2, &dev_data.dev_graph_index, dev_data.dev_graph_value, params->hidden_dim));

    // ReLU
    modules.push_back(std::make_unique<ReLU>(layer1_var2));

    // dropout
    modules.push_back(std::make_unique<Dropout>(layer1_var2, params->dropout_layer1, dev_rand_states));

    // dense matmul
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->output_dim));
    shared_ptr<Variable> layer2_var1 = variables.back();

    variables.push_back(std::make_shared<Variable>(params->hidden_dim * params->output_dim, true, dev_rand_states));
    shared_ptr<Variable> layer2_weight = variables.back();
    layer2_weight->glorot(params->hidden_dim, params->output_dim);
    // layer2_weight->set_value(0.5, streams[0]);

    modules.push_back(std::make_unique<Matmul>(layer1_var2, layer2_weight, layer2_var1, params->num_nodes, params->hidden_dim, params->output_dim));

    // graph sum
    variables.push_back(std::make_shared<Variable>(params->num_nodes * params->output_dim));
    output = variables.back();
    modules.push_back(std::make_unique<GraphSum>(layer2_var1, output, &dev_data.dev_graph_index, dev_data.dev_graph_value, params->output_dim));

    // cross entropy loss
    modules.push_back(std::make_unique<CrossEntropyLoss>(output, dev_truth, params->output_dim, dev_loss_train, dev_loss_eval));

    // optimizer
    optimizer = Adam({{layer1_weight, true}, {layer2_weight, false}}, adam_params);
}

// ##################################################################################

__global__ void initialize_random_kernel(randState *dev_rand_states)
{
    // curand_init(seed, index, offset, &state);
    curand_init(SEED * threadIdx.x, threadIdx.x, threadIdx.x, &dev_rand_states[threadIdx.x]);
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

#ifdef FEATURE
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
    const natural n_blocks = std::min(CEIL(input->size, N_THREADS), N_BLOCKS);
    if (!first)
        cudaStreamWaitEvent(stream.get(), events[2].get());
    set_input_kernel<<<n_blocks, N_THREADS, 0, stream.get()>>>(input->dev_data.get(), dev_data.dev_feature_value.get(), input->size);
    if (!first)
        cudaEventRecord(events[5].get(), stream.get());
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}
#endif

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

    const natural n_blocks = std::min(CEIL(params->num_nodes, N_THREADS), N_BLOCKS);
    cudaStreamWaitEvent(stream.get(), events[7].get());
    set_truth_kernel<<<n_blocks, N_THREADS, 0, stream.get()>>>(dev_truth.get(), dev_data.dev_split.get(), dev_data.dev_label.get(), params->num_nodes, current_split);
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

void GCN::get_l2_penalty(smart_stream stream, bool training) const
{
    dev_shared_ptr<real> dev_l2 = training ? dev_l2_train : dev_l2_eval;
    dev_l2.set_zero(stream);

    const auto &weights = variables[2];
    const natural n_blocks = std::min(CEIL(weights->size, N_THREADS), N_BLOCKS);

    get_l2_penalty_kernel<<<n_blocks, N_THREADS, 0, stream.get()>>>(dev_l2.get(), weights->dev_data.get(), weights->size);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// ##################################################################################

__global__ void get_accuracy_kernel(const integer *dev_truth, const real *dev_data, real *wrong, const natural num_nodes, const natural output_dim)
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

void GCN::get_accuracy(smart_stream stream, bool training) const
{
    dev_shared_ptr<real> dev_wrong = training ? dev_wrong_train : dev_wrong_eval;
    dev_wrong.set_zero(stream);
    const natural n_blocks = std::min(CEIL(params->num_nodes, N_THREADS), N_BLOCKS);
    get_accuracy_kernel<<<n_blocks, N_THREADS, 0, stream.get()>>>(dev_truth.get(), output->dev_data.get(), dev_wrong.get(), params->num_nodes, params->output_dim);
#ifdef DEBUG_CUDA
    CHECK_CUDA_ERROR(cudaGetLastError());
#endif
}

// ##################################################################################

__global__ void finalize_kernel(real *dev_l2, real *dev_loss, real *dev_wrong, const natural samples, const real weight_decay)
{
    if (threadIdx.x == 0)
    {
        *dev_loss /= samples;
        *dev_l2 = weight_decay * (*dev_l2) / real(2);
        *dev_loss = *dev_loss + *dev_l2;
    }
    if (threadIdx.x == 1)
        *dev_wrong = (samples - *dev_wrong) / static_cast<real>(samples);
}

__global__ void print(real *train_loss, real *train_acc, real *val_loss, real *val_acc, natural epoch)
{
    if (threadIdx.x == 0)
        printf("epoch=%d train_loss=%.5f train_acc=%.5f val_loss=%.5f val_acc=%.5f\n", epoch, *train_loss, *train_acc, *val_loss, *val_acc);
}

__global__ void print_test(real *test_loss, real *test_acc)
{
    if (threadIdx.x == 0)
        printf("test_loss=%.5f test_acc=%.5f\n", *test_loss, *test_acc);
}

// ##################################################################################

void GCN::eval(const natural current_split, const natural epoch) const
{
#ifdef FEATURE
    set_input(streams[3], false);
#endif

    for (natural i = 0; i < modules.size() - 1; i++)
        modules[i]->forward(false, streams[3]);
    set_truth(current_split, streams[3]);
    modules.back()->forward(false, streams[3]);

    get_l2_penalty(streams[3], false);
    get_accuracy(streams[3], false);
}

// ##################################################################################

void GCN::train_epoch()
{
    // input set by constructor and by eval()
    cudaStreamWaitEvent(streams[0].get(), events[5].get());

    for (natural i = 0; i < modules.size() - 1; i++)
        modules[i]->forward(true, streams[0]);
    set_truth(1, streams[0]); // get the true labels for the dataset with split == 1 (train)
    modules.back()->forward(true, streams[0]);

    get_l2_penalty(streams[0], true);
    get_accuracy(streams[0], true);
    finalize_kernel<<<1, 2, 0, streams[0].get()>>>(dev_l2_train.get(), dev_loss_train.get(), dev_wrong_train.get(), params->train_dim, adam_params->weight_decay);
    cudaEventRecord(events[13].get(), streams[0].get());

    for (integer i = modules.size() - 1; i >= 0; i--)
        modules[i]->backward(); // do a backward pass on the layers

    optimizer.step(); // apply a step of the adam optimization
}

// ##################################################################################

__global__ void early_stopping_kernel(real *dev_loss_history, const real *dev_loss_eval, const natural epoch, const natural early_stopping, natural *interrupt)
{
    if (threadIdx.x == 0)
    {
        dev_loss_history[epoch - 1] = *dev_loss_eval;
        if (epoch >= early_stopping)
        {
            real recent_loss = 0.0;
            for (natural i = epoch - early_stopping; i < epoch; i++)
                recent_loss += dev_loss_history[i];
            if (*dev_loss_eval > recent_loss / static_cast<real>(early_stopping))
            {
                printf("Early stopping...\n");
                *interrupt = 1;
            }
        }
    }
}

void GCN::run()
{
    timer_start(TMR_TOTAL);
    natural epoch = 1;
    // Iterate the training process based on the selected number of epoch
    for (; epoch <= params->epochs; epoch++)
    {

        // train the epoch and record the current train_loss and train_accuracy
        train_epoch();

        // eval the model at the current step in order to obtain the val_loss and val_accuracy
        eval(2, false);
        finalize_kernel<<<1, 2, 0, streams[3].get()>>>(dev_l2_eval.get(), dev_loss_eval.get(), dev_wrong_eval.get(), params->val_dim, adam_params->weight_decay);

        cudaStreamWaitEvent(streams[3].get(), events[13].get());
        print<<<1, 1, 0, streams[3].get()>>>(dev_loss_train.get(), dev_wrong_train.get(), dev_loss_eval.get(), dev_wrong_eval.get(), epoch);

        if (params->early_stopping > 0)
        {
            early_stopping_kernel<<<1, 1, 0, streams[3].get()>>>(dev_loss_history.get(), dev_loss_eval.get(), epoch, params->early_stopping, dev_interrupt.get());
            natural interrupt = 0;
            dev_interrupt.copy_to_host_async(&interrupt, streams[3]);
            cudaStreamSynchronize(streams[3].get());
            if (interrupt)
                break;
        }
    }

    cudaDeviceSynchronize();
    timer_stop(TMR_TOTAL);

    eval(3, false); // eval the model on the test set
    finalize_kernel<<<1, 2, 0, streams[3].get()>>>(dev_l2_eval.get(), dev_loss_eval.get(), dev_wrong_eval.get(), params->test_dim, adam_params->weight_decay);
    print_test<<<1, 1, 0, streams[3].get()>>>(dev_loss_eval.get(), dev_wrong_eval.get());

    std::cout << "total time: " << timer_total(TMR_TOTAL) << std::endl;
}

// ##################################################################################
