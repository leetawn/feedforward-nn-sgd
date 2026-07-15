#include "../include/activations.h"

void relu(Matrix *weighted_sums) {
    u32 n = weighted_sums->rows * weighted_sums->columns;
    for (u32 i = 0; i < n; i++) {
        weighted_sums->data[i] = fmax(weighted_sums->data[i], 0.0);
    }    
}

void relu_derivative(Matrix *weighted_sums) {
    u32 n = weighted_sums->rows * weighted_sums->columns;
    for (u32 i = 0; i < n; i++) {
        if (weighted_sums->data[i] > 0) weighted_sums->data[i] = 1.0;
        else weighted_sums->data[i] = 0.0;
    }
}

void softmax(Matrix *scores) {
    u32 n = scores->rows * scores->columns;
    float max_score = scores->data[0];

    for (u32 i = 0; i < n; i++) {
        max_score = fmaxf(max_score, scores->data[i]);
    }

    for (u32 i = 0; i < n; i++) {
        scores->data[i] -= max_score;
    }

    for (u32 i = 0; i < n; i++) {
        scores->data[i] = exp(scores->data[i]);
    }

    float exp_sum = 0.0;

    for (u32 i = 0; i < n; i++) {
        exp_sum += scores->data[i];
    }

    for (u32 i = 0; i < n; i++) {
        scores->data[i] /= exp_sum;
    }
}
