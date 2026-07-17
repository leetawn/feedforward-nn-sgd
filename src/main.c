#include "../include/config.h"
#include "../include/matrix.h"
#include "../include/mnist_loader.h"
#include "../include/activations.h"
#include "../include/network.h"
#include "../include/train.h"
#include "../include/image.h"

Matrix *get_image(MNIST_DS *dataset, u32 digit) {
    Matrix *image = allocate_matrix(784, 1);
    
    u32 *indices = (u32 *)malloc(sizeof(u32) * dataset->images);
    
    fy_shuffle(indices, dataset->images);

    for (u32 i = 0; i < dataset->images; i++) {
        u32 random_idx = indices[i];
        if (dataset->labels[random_idx] == digit) {
            memcpy(image->data, &dataset->pixels[random_idx * 784], sizeof(float) * 784);
            free(indices); 
            return image;
        }
    }

    printf("No image found with digit %d\n", digit);
    free(indices); 
    return image;
}
int main() {
    srand(time(NULL));
    char *test_images = "test/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
    char *test_labels = "test/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte";
    char *train_images = "train/train-images-idx3-ubyte/train-images.idx3-ubyte";
    char *train_labels = "train/train-labels-idx1-ubyte/train-labels.idx1-ubyte";

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

    // char op;
    // printf("Test on a single image file? (Y/N): ");
    // scanf(" %c", &op);
    // switch (op) {
    //     case 'Y': {
    //         u32 digit;
    //         printf("Enter digit to test: ");
    //         scanf("%u", &digit);

    //         Matrix *digit_image = get_image(test_set, digit);
    //         test_image(digit_image, network);
    //         free_matrix(digit_image);
    //         break;
    //     }

    //     case 'I': {
    //         char filename[1024];
    //         printf("Enter image filename: ");
    //         Matrix *image_matrix = load_image(filename);
    //         test_image(image_matrix, network);
    //         break;
    //     }

    //     case 'N': {
    //         printf("Exiting...\n");
    //         break;
    //     }
    // } 
    char filename[1024];
    printf("Enter image filename: ");
    scanf("%s", &filename);
    Matrix *image_matrix = load_image(filename);
    test_image(image_matrix, network);


    free(network);
    free(full_train_set);
    free(test_set);
    free(train_set);
    free(val_set);

    return 0;
}