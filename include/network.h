#pragma once
#include "../include/config.h"
#include "../include/matrix.h"
#include "../include/activations.h"

typedef enum {
    RELU,
    SOFTMAX
} ActivationType;

typedef struct Layer {
    Matrix *weight_matrix;
    Matrix *bias_vector;
    Matrix *pre_activation_vals;
    Matrix *post_activation_vals;
    Matrix *bias_gradient;
    Matrix *weight_gradient;
    ActivationType activation_type;
} Layer;


typedef struct Network {
    Layer *layers;
} Network;

void xavier_glorot_init(Matrix *weight_matrix, u32 input_neurons, u32 output_neurons);
void bias_init(Matrix *bias_vector);
Matrix *forward_pass(Matrix *input_image, Network *network);
Network *network_init();
void compute_weighted_sum(Matrix *input, Matrix *weighted_matrix, Matrix *bias_vector, Matrix *out);
void apply_activation(Matrix *input, Matrix *out, ActivationType activation);
void backpropagation(Network *network, Matrix *input_image, Matrix *onehot);
