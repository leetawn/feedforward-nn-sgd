#pragma once
#include "../include/config.h"

typedef struct MNIST_DS {
    float *pixels;
    u8 *labels;
    u32 images;
    u32 rows;
    u32 cols;
} MNIST_DS;

typedef struct Batch {
    float *pixels;
    u8 *labels;
    u32 size;
} Batch;

int parse_images(MNIST_DS *dataset, char *filename);
int parse_labels(MNIST_DS *dataset, char *filename);
float *normalize_data(u8 *data, u32 n);
MNIST_DS *load(char *image_path, char *label_path);
void fy_shuffle(u32 *indices, u32 n);
void split_train(MNIST_DS *full_train_set, MNIST_DS *out, MNIST_DS *val_set, u32 val_size);
Batch *get_batch(MNIST_DS *dataset, u32 *indices, u32 batch_index, u32 batch_size);
void free_batch(Batch *batch);