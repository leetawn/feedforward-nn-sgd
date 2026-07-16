#pragma once
#include "../include/config.h"
#include "../include/matrix.h"

void relu(Matrix *weighted_sums);
void relu_derivative(Matrix *weighted_sums);
void softmax(Matrix *scores);
