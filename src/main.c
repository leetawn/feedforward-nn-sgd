#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/config.h"
#include "../include/matrix.h"
#include "../include/mnist_loader.h"
#include "../include/activations.h"
#include "../include/network.h"
#include "../include/train.h"
#include "../include/stb_image_write.h"

Matrix *get_image(MNIST_DS *dataset, u32 digit) {
    Matrix *image = allocate_matrix(784, 1);
    for (u32 i = 0; i < dataset->images; i++) {
        if (dataset->labels[i] == digit) {
            memcpy(image->data, &dataset->pixels[i * 784], sizeof(float) * 784);
            return image;
        }
    }
    printf("No image found with digit %d\n", digit);
    return image;
}

void save_png(Matrix *image, const char *filename) {
    uc pixels[784];

    for (u32 i = 0; i < 784; i++) {
        float val = image->data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0;
        pixels[i] = (uc)(val * 255.0f);
    }
    int result = stbi_write_png(filename, 28, 28, 1, pixels, 28);
    if (!result) {
        printf("Failed to write PNG.\n");
    }
}

void test_image(Matrix *image, Network *network) {
    save_png(image, "../out/test.png");
    Matrix *a3 = forward_pass(image, network);
    float predicted = -1.0f;
    u32 predicted_index = -1;
    for (u32 i = 0; i < 10; i++) {
        printf("%d: %f\n", i, a3->data[i]);
        if (a3->data[i] > predicted) {
            predicted_index = i;
            predicted = a3->data[i];
        }
    }
    predicted *= 100;
    printf("Image shows a %d with %f%% confidence", predicted_index, predicted);
}
int main() {
    srand(time(NULL));
    char *test_images = "../test/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
    char *test_labels = "../test/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte";
    char *train_images = "../train/train-images-idx3-ubyte/train-images.idx3-ubyte";
    char *train_labels = "../train/train-labels-idx1-ubyte/train-labels.idx1-ubyte";

    MNIST_DS *full_train_set = load(train_images, train_labels);
    MNIST_DS *test_set = load(test_images, test_labels);

    MNIST_DS *train_set = (MNIST_DS *)malloc(sizeof(MNIST_DS));
    MNIST_DS *val_set = (MNIST_DS *)malloc(sizeof(MNIST_DS));

    split_train(full_train_set, train_set, val_set, VALIDATION_SET_SIZE);
    printf("Split into train/val sets.\n");

    Network *network = network_init();
    printf("Network initialized.\n");

    train(network, train_set, val_set, EPOCHS);
    printf("Training complete.\n");
    float final_eval = per_epoch_eval(test_set, network);

    printf("Final Test Accuracy: %f\n", final_eval);

    Matrix *digit_image = get_image(test_set, 1);
    test_image(digit_image, network);
    free_matrix(digit_image);

    free(network);
    free(full_train_set);
    free(test_set);
    free(train_set);
    free(val_set);

    return 0;
}