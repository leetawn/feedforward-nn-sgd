#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "../include/image.h"
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include "../include/stb_image_resize2.h"

void reverse_string(char *string, u32 n) {
    u32 l = 0, r = n - 1;
    while (l <= r) {
        char temp = string[l];
        string[l] = string[r];
        string[r] = temp;
        l++; r--;
    }
}

uc *resize_image(char *filename, uc *original_data, u32 original_width, u32 original_height, u32 original_channels) {
    u32 target_width = 28, target_height = 28; 
    size_t out_size = target_width * target_height;

    uc *output_data = (uc *)malloc(sizeof(uc) * out_size);

    uc *result = stbir_resize_uint8_linear(
        original_data, original_width, original_height, 0,
        output_data, target_width, target_height, 0,
        STBIR_1CHANNEL
    );

    if (!result) {
        printf("Resizing failed.\n");
        free(output_data);
        return NULL;
    } 
    stbi_write_png("test.png", target_width, target_height, 1, output_data, 0);
    return output_data;
}

Matrix *load_image(char *filename) {
    u32 width, height, channels;
    uc *data = stbi_load(filename, &width, &height, &channels, STBI_grey);
    if (!data) {
        printf("Error in loading image.\n");
        return NULL;
    }
    if (width != 28 || height != 28) {
        uc *resized_data = resize_image(filename, data, width, height, 1);
        stbi_image_free(data);
        data = resized_data;
        width = 28;
        height = 28;
    }

    float *normalized_data = normalize_data(data, width * height);
    Matrix *result = allocate_matrix(784, 1);
    memcpy(result->data, normalized_data, sizeof(float) * 784);
    
    free(data);
    free(normalized_data);
    return result;
}

IMAGE_EXTENSION get_image_extension(char *filename) {
    char buffer[1024];
    u32 buffer_index = 0;
    size_t filename_length = strlen(filename);    
    for (u32 i = filename_length - 1; i >= 0; i--) {
        buffer[buffer_index++] = filename[i];    
    }
    reverse_string(buffer, buffer_index);
    buffer[buffer_index++] = '\0';

    if (buffer == "JPG") return JPG;
    if (buffer == "JPEG") return JPEG;
    if (buffer == "PNG") return PNG;
    return INVALID;
}

void test_image(Matrix *image, Network *network) {
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
    printf("Image shows a %d with %f%% confidence\n\n", predicted_index, predicted);
}

/*
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
    } else {
        printf("Image generated in the out folder.\n");
    }
}
*/
