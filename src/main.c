#include "../include/config.h"
#include "../include/matrix.h"
#include "../include/mnist_loader.h"
#include "../include/activations.h"
#include "../include/network.h"
#include "../include/train.h"
#include "../include/image.h"
#include "../include/gui.h"
#include "../include/progress_bar.h"


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

    char op;
    int prompt = 1;
    while (prompt) {
        printf("\n\nDefault hyperparameter values are based on standard values.\nIt is recommended to keep the Epoch count between 10-20, Learning Rate at 0.01, and Validation Set Size at 5000.\nDo you want to set custom hyperparameter values? (y/n): ");
        scanf(" %c", &op);
        switch (op) {
            case 'y': {
                printf("Epochs: ");
                scanf("%d", &EPOCHS);
                printf("Learning Rate: ");
                scanf("%f", &LEARNING_RATE);
                printf("Validation Set Size: ");
                scanf("%d", &VALIDATION_SET_SIZE);
                prompt = 0;
                break;
            }

            case 'n': {
                printf("Using default values.\n");
                prompt = 0;
                break;
            }
            case 'x': {
                prompt = 0;
                break;
            }
            default: {
                printf("Invalid option.\n");
                break;
            }
        }
    }
    split_train(full_train_set, train_set, val_set, VALIDATION_SET_SIZE);

    Network *network = network_init();
    printf("Network initialized.\n");

    printf("Starting training.\n");
    train(network, train_set, val_set, EPOCHS);
    printf("Training complete.\n");
    float final_eval = per_epoch_eval(test_set, network);

    printf("Final Test Accuracy: %.4f%%\n", final_eval * 100.0);

    display(network); 


    free(network);
    free(full_train_set);
    free(test_set);
    free(train_set);
    free(val_set);

    return 0;
}