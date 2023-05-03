#include "../include/module.h"
#include "../include/rand.h"
#include "../include/timer.h"
#include <cmath>

// ################################################################################################################
/**
 * Dense matrix multiplication layer.
 */
Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p)
    : a(a), b(b), c(c), m(m), n(n), p(p) {}

void Matmul::forward(bool training) {
  timer_start(TMR_MATMUL_FW);
  c->zero();
  // the result c is an N x F matrix, i.e. m x p here
  for (int i = 0; i < m; i++) // iterate over rows of c
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < p; k++) // iterate over columns of c
        c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
    }
  timer_stop(TMR_MATMUL_FW);
}

void Matmul::backward() {
  timer_start(TMR_MATMUL_BW);
  a->zero_grad(); // N x H i.e. m x n
  b->zero_grad(); // H x F i.e. n x p
  // c is N x F i.e. m x p
  for (int i = 0; i < m; i++)     // iterate over rows of a
    for (int j = 0; j < n; j++) { // iterate over columns of a
      float tmp = 0;
      for (int k = 0; k < p; k++) {
        tmp += c->grad[i * p + k] * b->data[j * p + k]; // a = c * b^T
        b->grad[j * p + k] +=
            c->grad[i * p + k] * a->data[i * n + j]; // b = a^T * c
      }
      a->grad[i * n + j] = tmp;
    }
  timer_stop(TMR_MATMUL_BW);
}

// ################################################################################################################

/**
 * A sparse matrix multiplication layer.
 */
SparseMatmul::SparseMatmul(Variable *b, Variable *c, SparseIndex *sp, int m,
                           int n, int p)
    : b(b), c(c), sp(sp), m(m), n(n), p(p) {}

void SparseMatmul::forward(bool training) {
  timer_start(TMR_SPMATMUL_FW);
  c->zero();
  // store in c the result of a * b (layer1_var1 = X * W_0)
  for (int i = 0; i < sp->indptr.size() - 1;
       i++) // iterate over rows of c (X), "sp->indptr.size() - 1" is N
    for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1];
         jj++) {                  // iterate over columns of X
      int j = sp->indices[jj];    // get the column index of the i-th row of X
      for (int k = 0; k < p; k++) // iterate over columns of c (W_0)
        c->data[i * p + k] += b->data[j * p + k];
    }
  timer_stop(TMR_SPMATMUL_FW);
}

void SparseMatmul::backward() {
  timer_start(TMR_SPMATMUL_BW);
  b->zero_grad();
  int row = 0;
  for (int i = 0; i < sp->indptr.size() - 1; i++)
    for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
      int j = sp->indices[jj];
      for (int k = 0; k < p; k++)
        b->grad[j * p + k] += c->grad[i * p + k];
    }
  timer_stop(TMR_SPMATMUL_BW);
}

// ################################################################################################################

/**
 * A specialized sparse matrix multiplication for graphs.
 */
GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim)
    : in(in), out(out), graph(graph), dim(dim) {}

void GraphSum::forward(bool training) {
  timer_start(TMR_GRAPHSUM_FW);
  out->zero();
  for (int src = 0; src < graph->indptr.size() - 1; src++)
    // "graph->indptr.size() - 1" is N
    for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
      int dst = graph->indices[i];
      // Can I store values of A^ outside ???
      float coef = 1.0 / sqrtf((graph->indptr[src + 1] - graph->indptr[src]) *
                               (graph->indptr[dst + 1] - graph->indptr[dst]));
      for (int j = 0; j < dim; j++)
        // This only works for undirected graphs. Should be out[dst] += coef *
        // in[src]
        out->data[src * dim + j] += coef * in->data[dst * dim + j];
    }
  timer_stop(TMR_GRAPHSUM_FW);
}

void GraphSum::backward() {
  timer_start(TMR_GRAPHSUM_BW);
  in->zero_grad();
  for (int src = 0; src < graph->indptr.size() - 1; src++)
    for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
      int dst = graph->indices[i];
      float coef = 1.0 / sqrtf((graph->indptr[src + 1] - graph->indptr[src]) *
                               (graph->indptr[dst + 1] - graph->indptr[dst]));
      for (int j = 0; j < dim; j++)
        in->grad[src * dim + j] += coef * out->grad[dst * dim + j];
    }
  timer_stop(TMR_GRAPHSUM_BW);
}

// ################################################################################################################

/**
 * Each predicted class probability is compared to the actual class desired and
 * a loss is computed to penalize the proabability based on how far it is with
 * respect to the actual expected value. Also called logaritmic loss.
 */
CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss,
                                   int num_classes)
    : logits(logits), truth(truth), loss(loss), num_classes(num_classes) {}

void CrossEntropyLoss::forward(bool training) {
  timer_start(TMR_LOSS_FW);
  float total_loss = 0;
  int count = 0;
  if (training)
    logits->zero_grad();

  // iterate over outputs rows (nodes)
  for (int i = 0; i < logits->data.size() / num_classes; i++) {
    if (truth[i] < 0) // only train labels
      continue;
    count++; // count the number of training nodes (labels)
    float *logit =
        &logits->data[i * num_classes]; // output row of the i-th node
    float max_logit = -1e30, sum_exp = 0;
    // get the maximum value of each node
    for (int j = 0; j < num_classes; j++)
      max_logit = fmax(max_logit, logit[j]);
    // calculate the loss applying the softmax
    for (int j = 0; j < num_classes; j++) {
      logit[j] -= max_logit; // numerical stability
      sum_exp += expf(logit[j]);
    }
    total_loss += logf(sum_exp) - logit[truth[i]]; // eq (10) of the paper ???

    if (training) {
      for (int j = 0; j < num_classes; j++) {
        float prob = expf(logit[j]) / sum_exp;
        logits->grad[i * num_classes + j] = prob;
      }
      logits->grad[i * num_classes + truth[i]] -= 1.0;
    }
  }

  *loss = total_loss / count;
  if (training)
    for (float &i : logits->grad)
      i /= count;
  timer_stop(TMR_LOSS_FW);
}

void CrossEntropyLoss::backward() {}

// ################################################################################################################

/**
 * Rectified Linear Unit activation function.
 * If input is negative it will output 0.
 */
ReLU::ReLU(Variable *in) {
  this->in = in;
  mask = new bool[in->data.size()];
}

ReLU::~ReLU() { delete[] mask; }

void ReLU::forward(bool training) {
  timer_start(TMR_RELU_FW);
  for (int i = 0; i < in->data.size(); i++) {
    bool keep = in->data[i] > 0;
    if (training)
      mask[i] = keep;
    if (!keep)
      in->data[i] = 0;
  }
  timer_stop(TMR_RELU_FW);
}

void ReLU::backward() {
  timer_start(TMR_RELU_BW);
  for (int i = 0; i < in->data.size(); i++)
    if (!mask[i])
      in->grad[i] = 0;
  timer_stop(TMR_RELU_BW);
}

// ################################################################################################################

/**
 * The dropout layer randomly sets input units to 0 with a frequency of P at
 * each step during training time to prevent overfitting. Inputs that are not
 * set to 0 are scaled up by 1/(1-P).
 */
Dropout::Dropout(Variable *in, float p) {
  this->in = in;
  this->p = p;
  if (!in->grad.empty())
    mask = new int[in->data.size()];
  else
    mask = nullptr;
}

Dropout::~Dropout() {
  if (mask)
    delete[] mask;
}

void Dropout::forward(bool training) {
  if (!training)
    return;
  timer_start(TMR_DROPOUT_FW);
  const int threshold = int(p * MY_RAND_MAX);
  float scale = 1 / (1 - p);
  for (int i = 0; i < in->data.size(); i++) {
    bool keep = (int)RAND() >= threshold;
    in->data[i] *= keep ? scale : 0;
    if (mask)
      mask[i] = keep; // to optimize: check outside the loop
  }
  timer_stop(TMR_DROPOUT_FW);
}

void Dropout::backward() {
  if (!mask)
    return;
  timer_start(TMR_DROPOUT_BW);
  float scale = 1 / (1 - p);
  for (int i = 0; i < in->data.size(); i++)
    in->grad[i] *= mask[i] ? scale : 0;
  timer_stop(TMR_DROPOUT_BW);
}

// ################################################################################################################