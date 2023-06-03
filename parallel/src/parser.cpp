#include "../include/parser.h"

Parser::Parser(GCNParams *gcnParams, GCNData *gcnData, std::string graph_name) {
  this->graph_file.open("data/" + graph_name + ".graph");
  this->split_file.open("data/" + graph_name + ".split");
  this->svmlight_file.open("data/" + graph_name + ".svmlight");
  this->gcnParams = gcnParams;
  this->gcnData = gcnData;
}

/**
 * By parsing the .graph file containing the ordered edgelist of the graph it
 * populates a CSR representation of the graph
 */
void Parser::parseGraph() {
  auto &graph =
      this->gcnData->graph; // Reference to "graph" member (SparseIndex)

  graph.indptr.push_back(0);
  int node = 0;
  // Iterate over the nodes
  while (true) {
    std::string line;
    getline(graph_file, line);
    if (graph_file.eof())
      break;

    // Implicit self connection (node X is connected with node X)
    graph.indices.push_back(node);
    // add the value that cumultatively counts the number of row elements
    graph.indptr.push_back(graph.indptr.back() + 1);
    node++;

    // iterate over the neighbours of the current node in order to populate the
    // CSR
    std::istringstream ss(line);
    while (true) {
      int neighbor;
      ss >> neighbor;
      if (ss.fail())
        break;
      graph.indices.push_back(neighbor);
      graph.indptr.back() += 1;
    }
  }

  gcnParams->num_nodes = node;
}

bool Parser::isValidInput() {
  return graph_file.is_open() && split_file.is_open() &&
         svmlight_file.is_open();
}

/**
 * The SVMLight allows for storing features in a sparse way therefore even the
 * features are stored using a CSR approach to index the data in feature_val
 */
void Parser::parseNode() {
  auto &feature_sparse_index =
      this->gcnData
          ->feature_index; // Reference to "feature_index" member (SparseIndex)
  auto &feature_val = this->gcnData->feature_value; // std::vector<float>
  auto &labels = this->gcnData->label;              // std::vector<int>

  feature_sparse_index.indptr.push_back(0);

  int max_idx = 0, max_label = 0;
  // Iterate over the nodes
  while (true) {
    std::string line;
    getline(svmlight_file, line);
    if (svmlight_file.eof())
      break;
    // add the value that cumultatively counts the number of row elements
    feature_sparse_index.indptr.push_back(feature_sparse_index.indptr.back());
    std::istringstream ss(line);

    // the first number in the svmlight format is the label of the node
    int label = -1;
    ss >> label;
    labels.push_back(label);
    if (ss.fail())
      continue;
    max_label = std::max(max_label, label);

    // after the label of the node we find a list of feature_id:feature_val
    // iterate over "feature_id:feature_val" blocks
    while (true) {
      std::string kv;
      ss >> kv;
      if (ss.fail())
        break;
      std::istringstream kv_ss(kv);

      int k;
      float v;
      char col;
      kv_ss >> k >> col >> v;
#ifdef NO_FEATURE
      feature_val.push_back(1.0);
#else
      feature_val.push_back(v);
#endif
      feature_sparse_index.indices.push_back(k);
      feature_sparse_index.indptr.back() += 1;
      max_idx = std::max(max_idx, k);
    }
  }
  gcnParams->input_dim = max_idx + 1;
  gcnParams->output_dim = max_label + 1;
}

void Parser::parseSplit() {
  auto &split = this->gcnData->split;

  while (true) {
    std::string line;
    getline(split_file, line);
    if (split_file.eof())
      break;
    int val = std::stoi(line);
    if (val == 1)
      gcnParams->train_dim++;
    else if (val == 2)
      gcnParams->val_dim++;
    else if (val == 3)
      gcnParams->test_dim++;

    split.push_back(val);
  }
}

// Function to calculate the local clustering coefficient for a given node
real Parser::local_clustering_coefficient(natural node) {
  auto &rowPtr = this->gcnData->graph.indptr;
  auto &colIndex = this->gcnData->graph.indices;
  const natural start = rowPtr[node];
  const natural end = rowPtr[node + 1];
  const natural edges = end - start;
  natural triangles = 0;
  real clusteringCoefficient = 0.0;
  // Count the number of edges and triangles for the node
  if (edges <= 1)
    return 0.0;

  for (natural i = start; i < end; ++i) {
    natural neighbor = colIndex[i];
    for (natural j = rowPtr[neighbor]; j < rowPtr[neighbor + 1]; ++j) {
      natural neighborOfNeighbor = colIndex[j];
      if (neighborOfNeighbor == node)
        continue;
      bool finded = false;
      for (natural k = start; k < end && !finded; ++k)
        if (colIndex[k] == neighborOfNeighbor) {
          triangles++;
          finded = true;
        }
    }
  }
  return static_cast<real>(triangles) / (edges * (edges - 1));
}

void Parser::calculateGraphValues() {
  auto &graph_idx = this->gcnData->graph;
  auto &graph_val = this->gcnData->graph_value;

  graph_val.resize(graph_idx.indices.size(), 1.);
  // iterate over the nodes
  natural temp = 0;
  for (int src = 0; src < graph_idx.indptr.size() - 1; src++) {
    for (int i = graph_idx.indptr[src]; i < graph_idx.indptr[src + 1]; i++) {
      int dst = graph_idx.indices[i];
      if (src == dst)
        graph_val[i] += 1; // graph_val[i] += local_clustering_coefficient[src];
      graph_val[i] /=
          sqrtf((graph_idx.indptr[src + 1] - graph_idx.indptr[src]) *
                (graph_idx.indptr[dst + 1] - graph_idx.indptr[dst]));
    }
  }
}

void vprint(std::vector<int> v) {
  for (int i : v)
    printf("%i ", i);
  printf("\n");
}

bool Parser::parse() {
  if (!isValidInput())
    return false;
#ifndef TUNE_CUDA
  std::cout << "PARSING DATA ..." << std::endl;
#endif
  this->parseGraph();
#ifndef TUNE_CUDA
  std::cout << "Parse Graph Succeeded." << std::endl;
#endif
  this->parseNode();
#ifndef TUNE_CUDA
  std::cout << "Parse Node Succeeded." << std::endl;
#endif
  this->parseSplit();
#ifndef TUNE_CUDA
  std::cout << "Parse Split Succeeded." << std::endl;
#endif
  this->calculateGraphValues();
  return true;
}

void parse_parameters(GetPot &datafile, GCNParams &params,
                      AdamParams &adam_params, bool print) {
  // GCNParams
  params.n_layers = datafile("n_layers", 0);
  params.hidden_dims = string2vec<natural>(datafile("hidden_dims", ""));
  if (params.hidden_dims.size() != params.n_layers - 1) {
    std::cerr << "Number of hidden dimensions must be 1 - n_layers"
              << std::endl;
    exit(1);
  }
  params.dropouts = string2vec<real>(datafile("dropouts", ""));
  if (params.dropouts.size() != params.n_layers) {
    std::cerr << "Number of dropouts must match number of layers" << std::endl;
    exit(1);
  }
  params.epochs = datafile("epochs", 0);
  params.early_stopping = datafile("early_stopping", 0);

  // AdamParams
  adam_params.learning_rate = datafile("learning_rate", 0.0);
  adam_params.weight_decay = datafile("weight_decay", 0.0);
  adam_params.beta1 = datafile("beta1", 0.0);
  adam_params.beta2 = datafile("beta2", 0.0);
  adam_params.eps = datafile("eps", 0.0);

  // CudaParams
  int dev;
  cudaDeviceProp devProp;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&devProp, dev);
  CudaParams::N_BLOCKS =
      datafile("num_blocks_factor", 0) * devProp.multiProcessorCount;
  CudaParams::N_THREADS = datafile("num_threads", 0);

  if (print) {
    std::cout << "PARSED PARAMETERS FROM GETPOT" << std::endl;
    std::cout << "n_layers: " << params.n_layers << std::endl;
    std::cout << "hidden_dims: ";
    for (auto i : params.hidden_dims)
      std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "dropouts: ";
    for (auto i : params.dropouts)
      std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "epochs: " << params.epochs << std::endl;
    std::cout << "early_stopping: " << params.early_stopping << std::endl;
    std::cout << "learning_rate: " << adam_params.learning_rate << std::endl;
    std::cout << "weight_decay: " << adam_params.weight_decay << std::endl;
    std::cout << "beta1: " << adam_params.beta1 << std::endl;
    std::cout << "beta2: " << adam_params.beta2 << std::endl;
    std::cout << "eps: " << adam_params.eps << std::endl;
    std::cout << "num_blocks: " << CudaParams::N_BLOCKS << std::endl;
    std::cout << "num_threads: " << CudaParams::N_THREADS << std::endl;
    std::cout << std::endl;
  }
}
