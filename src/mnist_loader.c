#include "../include/mnist_loader.h"

void toggle_endian(u32 *number) {
    u32 t = 0;

    u32 b1 = *number & 0x000000FF;
    u32 b2 = (*number & 0x0000FF00) >> 8;
    u32 b3 = (*number & 0x00FF0000) >> 16;
    u32 b4 = (*number & 0xFF000000) >> 24;

    t |= b4;
    t |= (b3) << 8;
    t |= (b2) << 16;
    t |= (b1) << 24;

    *number = t;
}


float *normalize_data(u8 *data, u32 n) {
    float *normalized = (float *)malloc(sizeof(float) * n);
    for (u32 i = 0; i < n; i++) {
        float current = (float)(data[i]) / 255.0f;
        normalized[i] = current;
    }
    return normalized;
}

void initialize_indices(u32 *indices, u32 n) {
    for (u32 i = 0; i < n; i++) indices[i] = i;
}

int parse_images(MNIST_DS *dataset, char *filename) {
    FILE *data = fopen(filename, "rb");

    if (!data) {
        printf("File not opened.\n");
        return -1;
    }

    u32 magic = 0, remain = 0;

    fread(&magic, sizeof(u32), 1, data);
    toggle_endian(&magic);
    
    fread(&remain, sizeof(u32), 1, data);
    toggle_endian(&remain);
    

    if (magic != 2051) {
        printf("Magic number does not match the correct magic number. Expected 2051, got: %d\n", magic);
        return -1;
    }
    u32 rows = 0, cols = 0;
    fread(&rows, sizeof(u32), 1, data);
    fread(&cols, sizeof(u32), 1, data);
    toggle_endian(&rows); toggle_endian(&cols);

    u32 total = remain * rows * cols;

    u8 *loaded_data = (u8 *)malloc(sizeof(u8) * total);
    fread(loaded_data, sizeof(u8), total, data);


    fclose(data);
    float *normalized_data = normalize_data(loaded_data, total);

    dataset->pixels = normalized_data;
    dataset->images = remain;
    dataset->rows = rows;
    dataset->cols = cols;

    free(loaded_data);
    return 0;
}

int parse_labels(MNIST_DS *dataset, char *filename) {
    FILE *data = fopen(filename, "rb");

    if (!data) {
        printf("File not opened.\n");
        return -1;
    }

    u32 magic = 0, remain = 0;

    fread(&magic, sizeof(u32), 1, data);
    toggle_endian(&magic);
    
    fread(&remain, sizeof(u32), 1, data);
    toggle_endian(&remain);
    

    if (magic != 2049) {
        printf("Magic number does not match the correct magic number. Expected 2049, got: %d\n", magic);
        return -1;
    }

    u8 *labels = (u8 *)malloc(sizeof(u8) * remain);
    fread(labels, sizeof(u8), remain, data);

    dataset->labels = labels;

    fclose(data);
    return 0;
}

MNIST_DS *load(char *image_path, char *label_path) {
    MNIST_DS *dataset = (MNIST_DS *)malloc(sizeof(MNIST_DS));

    if (parse_images(dataset, image_path) == -1) {
        printf("Image parsing failed.\n");
        return NULL;
    }

    if (parse_labels(dataset, label_path) == -1) {
        printf("Label parsing failed.\n");
        return NULL;
    }

    return dataset;
}

void fy_shuffle(u32 *indices, u32 n) {
    // Durstenfeld's Implementation of Fisher-Yates
    initialize_indices(indices, n);

    for (u32 i = n - 1; i > 0; i--) {
        u32 k = rand() % (i + 1);

        u32 t = indices[i];
        indices[i] = indices[k];
        indices[k] = t;
    }
}

void split_train(MNIST_DS *full_train_set, MNIST_DS *train_set, MNIST_DS *val_set, u32 val_size) {
    u32 n = full_train_set->images;
    u32 *indices = (u32 *)malloc(sizeof(u32) * n);
    fy_shuffle(indices, n);

    u32 train_set_size = n - val_size;
    u32 offset = full_train_set->rows * full_train_set->cols;

    train_set->images = train_set_size;
    train_set->rows = full_train_set->rows;
    train_set->cols = full_train_set->cols;
    train_set->pixels = (float *)malloc(sizeof(float) * (train_set_size * offset));
    train_set->labels = (u8 *)malloc(sizeof(u8) * train_set_size);

    val_set->images = val_size;
    val_set->rows = full_train_set->rows;
    val_set->cols = full_train_set->cols;
    val_set->pixels = (float *)malloc(sizeof(float) * (val_size * offset));
    val_set->labels = (u8 *)malloc(sizeof(u8) * val_size);


    for (u32 i = 0; i < train_set_size; i++) {
        memcpy(&train_set->pixels[i * offset], &full_train_set->pixels[indices[i] * offset], offset * sizeof(float));
        train_set->labels[i] = full_train_set->labels[indices[i]]; 
    } 

    for (u32 i = 0; i < val_size; i++) {
        memcpy(&val_set->pixels[i * offset], &full_train_set->pixels[indices[train_set_size + i] * offset], offset * sizeof(float));
        val_set->labels[i] = full_train_set->labels[indices[train_set_size + i]];
    }
    
    free(indices);
}

Batch *get_batch(MNIST_DS *dataset, u32 *indices, u32 batch_index, u32 batch_size) {
    u32 offset = dataset->rows * dataset->cols;

    u32 remaining = dataset->images - batch_index;
    u32 size = (remaining < batch_size) ? remaining : batch_size;

    Batch *batch = (Batch *)malloc(sizeof(Batch));
    batch->size = size;
    batch->pixels = (float *)malloc(sizeof(float) * offset * size);
    batch->labels = (u8 *)malloc(sizeof(u8) * size);
    for (u32 i = 0; i < size; i++) {
        u32 src = indices[batch_index + i];
        memcpy(&batch->pixels[i * offset], &dataset->pixels[src * offset], offset * sizeof(float));
        batch->labels[i] = dataset->labels[src];
    }

    return batch;
}

void free_batch(Batch *batch) {
    free(batch->pixels);
    free(batch->labels);
    free(batch);
}
