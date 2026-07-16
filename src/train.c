#include "../include/train.h"

void onehot(u32 label, Matrix *out) {
    for (u32 i = 0; i < out->rows; i++) out->data[i] = 0.0;
    out->data[label] = 1.0;
}

float cross_entropy(Matrix *a3, u32 label) {
    float epsilon = 1e-7;
    float predicted = a3->data[label];
    if (predicted < epsilon) predicted = epsilon;
    return -log(predicted);
}

void sgd_update(Network *network) {
    for (u32 i = 0; i < 3; i++) {
        Matrix *t_wg = allocate_matrix(network->layers[i].weight_gradient->rows, network->layers[i].weight_gradient->columns);
        memcpy(t_wg->data, network->layers[i].weight_gradient->data, sizeof(float) * network->layers[i].weight_gradient->rows * network->layers[i].weight_gradient->columns);
        scalar_mult(t_wg, LEARNING_RATE); 
        int s = element_wise_sub(network->layers[i].weight_matrix, t_wg);
        if (s == 0) {
            printf("Error in updating weight values.\n");
            return;
        }
        Matrix *t_bg = allocate_matrix(network->layers[i].bias_gradient->rows, network->layers[i].bias_gradient->columns);
        memcpy(t_bg->data, network->layers[i].bias_gradient->data, sizeof(float) * network->layers[i].bias_gradient->rows * network->layers[i].bias_gradient->columns);
        scalar_mult(t_bg, LEARNING_RATE); 
        int vs = element_wise_sub(network->layers[i].bias_vector, t_bg);
        if (vs == 0) {
            printf("Error in updating bias values.\n");
            return;
        }
        free_matrix(t_wg);
        free_matrix(t_bg);
    }
}

u32 argmax(Matrix *v) {
    u32 max_index = 0;
    float max_value = v->data[0];
    for (u32 i = 1; i < v->rows; i++) {
        if (v->data[i] > max_value) {
            max_value = v->data[i];
            max_index = i;
        }
    }
    return max_index;
}

float per_epoch_eval(MNIST_DS *val_set, Network *network) {
    u32 correct = 0;
    Matrix *input_image = allocate_matrix(784, 1);

    for (u32 image_index = 0; image_index < val_set->images; image_index++) {
        ptom(val_set, image_index, input_image);
        Matrix *a3 = forward_pass(input_image, network);

        u32 predicted_label = argmax(a3);
        if (predicted_label == val_set->labels[image_index]) correct++;
    }
    free_matrix(input_image);
    return (float)correct / val_set->images;
}


void ptom(MNIST_DS *dataset, u32 image_index, Matrix *out) {
    u32 offset = dataset->rows * dataset->cols;
    memcpy(out->data, &dataset->pixels[image_index * offset], sizeof(float) * offset);
    out->rows = offset;
    out->columns = 1;
}

void train(Network *network, MNIST_DS *train_set, MNIST_DS *val_set, u32 epochs) {
    u32 n = train_set->images;
    Matrix *input_image = allocate_matrix(784, 1);
    Matrix *onehot_vector = allocate_matrix(10, 1);
    u32 *indices = (u32 *)malloc(sizeof(u32) * n);

    for (u32 i = 0; i < epochs; i++) {
        fy_shuffle(indices, n);
        float total_loss = 0;

        for (u32 image_index = 0; image_index < train_set->images; image_index++) {
            ptom(train_set, indices[image_index], input_image);
            onehot(train_set->labels[indices[image_index]], onehot_vector);

            Matrix *a3 = forward_pass(input_image, network);
            total_loss += cross_entropy(a3, train_set->labels[indices[image_index]]);

            backpropagation(network, input_image, onehot_vector);
            sgd_update(network);    

            if (image_index % 1000 == 0) {
                printf("  Epoch %d - image %d/%d\n", i + 1, image_index, train_set->images);
                fflush(stdout);
            }
        }

        float average_loss = total_loss / train_set->images;
        float accuracy = per_epoch_eval(val_set, network);

        printf("\n----------------------------------\n");
        printf("Epoch %d:\nTotal Loss: %f\nAverage Loss: %f\nAccuracy: %f", i + 1, total_loss, average_loss, accuracy);
        printf("\n----------------------------------\n");
    }
    free(indices);
    free_matrix(input_image);
    free_matrix(onehot_vector);
}