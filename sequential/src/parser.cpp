#include "../include/parser.h"
#include <algorithm>
#include <sstream>

using namespace std;
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
  auto &graph_sparse_index =
      this->gcnData->graph; // Reference to "graph" member (SparseIndex)

  graph_sparse_index.indptr.push_back(0);
  int node = 0;
  // Iterate over the nodes
  while (true) {
    std::string line;
    getline(graph_file, line);
    if (graph_file.eof())
      break;

    // Implicit self connection (node X is connected with node X)
    graph_sparse_index.indices.push_back(node);
    // add the value that cumultatively counts the number of row elements
    graph_sparse_index.indptr.push_back(graph_sparse_index.indptr.back() + 1);
    node++;

    // iterate over the neighbours of the current node in order to populate the
    // CSR
    std::istringstream ss(line);
    while (true) {
      int neighbor;
      ss >> neighbor;
      if (ss.fail())
        break;
      graph_sparse_index.indices.push_back(neighbor);
      graph_sparse_index.indptr.back() += 1;
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
    max_label = max(max_label, label);

    // after the label of the node we find a list of feature_id:feature_val
    // iterate over "feature_id:feature_val" blocks
    while (true) {
      string kv;
      ss >> kv;
      if (ss.fail())
        break;
      std::istringstream kv_ss(kv);

      int k;
      float v;
      char col;
      kv_ss >> k >> col >> v;

      feature_sparse_index.indices.push_back(k);
      feature_sparse_index.indptr.back() += 1;
      max_idx = max(max_idx, k);
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
    split.push_back(std::stoi(line));
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
  this->parseGraph();
  std::cout << "Parse Graph Succeeded." << endl;
  this->parseNode();
  std::cout << "Parse Node Succeeded." << endl;
  this->parseSplit();
  std::cout << "Parse Split Succeeded." << endl;
  return true;
}
