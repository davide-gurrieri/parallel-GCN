#ifndef PARALLEL_GCN_PARSER_H
#define PARALLEL_GCN_PARSER_H

#include "../include/GetPot"
#include "../include/gcn.cuh"
#include "../include/sparse.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

class Parser {
public:
  Parser(GCNParams *gcnParams, GCNData *gcnData, std::string graph_name);
  bool parse();

private:
  std::ifstream graph_file;
  std::ifstream split_file;
  std::ifstream svmlight_file;
  GCNParams *gcnParams;
  GCNData *gcnData;
  void parseGraph();
  void parseNode();
  void parseSplit();
  void calculateGraphValues();
  bool isValidInput();
};

void parse_parameters(GetPot &datafile, GCNParams &params,
                      AdamParams &adam_params, bool print = false);

#endif // PARALLEL_GCN_PARSER_H
