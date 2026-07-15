#include "../include/network.h"

void layer_init(Layer *layer, u32 rows, u32 columns, ActivationType activation) {
    layer->weight_matrix = allocate_matrix(rows, columns);
    layer->bias_vector = allocate_matrix(rows, 1);

    xavier_glorot_init(layer->weight_matrix, columns, rows);
    bias_init(layer->bias_vector);

    layer->pre_activation_vals = allocate_matrix(rows, 1); 
    layer->post_activation_vals = allocate_matrix(rows, 1);
    layer->activation_type = activation;
}

void xavier_glorot_init(Matrix *weight_matrix, u32 input_neurons, u32 output_neurons) {
    // Uniform Xavier Initialization    
    float x = sqrt( 6.0 / (input_neurons + output_neurons) );
    for (u32 i = 0; i < weight_matrix->rows; i++) {
        for (u32 j = 0; j < weight_matrix->columns; j++) {  
            weight_matrix->data[i * weight_matrix->columns + j] = ((float)rand() / RAND_MAX) * (2.0 * x) - x;
        }
    }
}

void bias_init(Matrix *bias_vector) {
    for (u32 i = 0; i < bias_vector->rows; i++) {
        for (u32 j = 0; j < bias_vector->columns; j++) {
            bias_vector->data[i * bias_vector->columns + j] = 0;
        }
    } 
}

Network *network_init() {
    Network *network = (Network *)malloc(sizeof(Network));

    // 4 Layers: 784 -> 128 -> 64 -> 10
    int n = 3;
    network->layers = (Layer *)malloc(sizeof(Layer) * n);

    layer_init(&network->layers[0], 128, 784, RELU); 
    layer_init(&network->layers[1], 64, 128, RELU); 
    layer_init(&network->layers[2], 10, 64, SOFTMAX); 

    return network;
}

void compute_weighted_sum(Matrix *input, Matrix *weighted_matrix, Matrix *bias_vector, Matrix *out) {
    int t = mv_mult(weighted_matrix, input, out);
    if (t == 0) {
        printf("Error in multiplying input and weighted matrix.\n");
        return;
    }    

    int k = element_wise_add(out, bias_vector);
    if (k == 0) {
        printf("Error in adding bias vector.\n");
        return;
    }
}

void apply_activation(Matrix *input, Matrix *out, ActivationType activation) {
    memcpy(out->data, input->data, sizeof(float) * input->rows);
    if (activation == RELU) relu(out);    
    else softmax(out);
}

Matrix *forward_pass(Matrix *input_image, Network *network) {

    compute_weighted_sum(input_image, network->layers[0].weight_matrix, network->layers[0].bias_vector, network->layers[0].pre_activation_vals);    
    apply_activation(network->layers[0].pre_activation_vals, network->layers[0].post_activation_vals, network->layers[0].activation_type);
    input_image = network->layers[0].post_activation_vals;

    compute_weighted_sum(input_image, network->layers[1].weight_matrix, network->layers[1].bias_vector, network->layers[1].pre_activation_vals);    
    apply_activation(network->layers[1].pre_activation_vals, network->layers[1].post_activation_vals, network->layers[1].activation_type);
    input_image = network->layers[1].post_activation_vals;

    compute_weighted_sum(input_image, network->layers[2].weight_matrix, network->layers[2].bias_vector, network->layers[2].pre_activation_vals);    
    apply_activation(network->layers[2].pre_activation_vals, network->layers[2].post_activation_vals, network->layers[2].activation_type);
    input_image = network->layers[2].post_activation_vals;

    return input_image;
}


