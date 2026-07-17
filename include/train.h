#pragma once
#include "../include/config.h"
#include "../include/matrix.h"
#include "../include/network.h"
#include "../include/mnist_loader.h"
#include "../include/progress_bar.h"

void onehot(u32 label, Matrix *out);
float cross_entropy(Matrix *a3, u32 label);
void sgd_update(Network *network);
u32 argmax(Matrix *v);
float per_epoch_eval(MNIST_DS *val_set, Network *network);
void ptom(MNIST_DS *dataset, u32 image_index, Matrix *out);
void train(Network *network, MNIST_DS *train_set, MNIST_DS *val_set, u32 epochs);