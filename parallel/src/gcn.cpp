#include "../include/gcn.h"
//#include "../include/rand.h"
//#include "../include/timer.h"
#include <cstdio>
#include <tuple>

/**
 * Returns the default paramets of the model
 * they will be overwritten by the parser when reading the dataset
 */
GCNParams GCNParams::get_default() {
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
