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
#ifdef FEATURE
  auto &feature_val = this->gcnData->feature_value; // std::vector<float>
#endif
  auto &labels = this->gcnData->label; // std::vector<int>

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
#ifdef FEATURE
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

void Parser::calculateGraphValues() {
  auto &graph_idx = this->gcnData->graph;
  auto &graph_val = this->gcnData->graph_value;

  graph_val.resize(graph_idx.indices.size());
  // iterate over the nodes
  for (int src = 0; src < graph_idx.indptr.size() - 1; src++) {
    for (int i = graph_idx.indptr[src]; i < graph_idx.indptr[src + 1]; i++) {
      int dst = graph_idx.indices[i];
      graph_val[i] =
          1.0 / sqrtf((graph_idx.indptr[src + 1] - graph_idx.indptr[src]) *
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
#ifndef TUNE
  std::cout << "PARSING DATA ..." << std::endl;
#endif
  this->parseGraph();
#ifndef TUNE
  std::cout << "Parse Graph Succeeded." << std::endl;
#endif
  this->parseNode();
#ifndef TUNE
  std::cout << "Parse Node Succeeded." << std::endl;
#endif
  this->parseSplit();
#ifndef TUNE
  std::cout << "Parse Split Succeeded." << std::endl;
#endif
  this->calculateGraphValues();
  return true;
}
