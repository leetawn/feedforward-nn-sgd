#pragma once
#include "../include/config.h"
#include "../include/matrix.h"
#include "../include/mnist_loader.h"
#include "../include/network.h"

typedef enum IMAGE_EXTENSION {
    INVALID = -1,
    JPG,
    JPEG,
    PNG,
} IMAGE_EXTENSION;

Matrix *load_image(char *filename);
uc *resize_image(char *filename, uc *original_data, u32 original_width, u32 original_height, u32 original_channels);
IMAGE_EXTENSION get_image_extension(char *filename);
// void save_image(Matrix *image, char *out);
void test_image(Matrix *image, Network *network);