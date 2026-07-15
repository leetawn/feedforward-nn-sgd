#include "../include/matrix.h"
#include <stdlib.h>
#include <stdio.h>


Matrix *allocate_matrix(u32 rows, u32 columns) {
    Matrix *in = (Matrix *)malloc(sizeof(Matrix));
    in->rows = rows;
    in->columns = columns;
    in->data = (float *)calloc((rows * columns), sizeof(float));

    return in;
}

void free_matrix(Matrix *in) {
    free(in->data);
    free(in);
}

int multiply(Matrix *a, Matrix *b, Matrix *out) {
    if (a->columns != b->rows) return 0;

    for (int i = 0; i < out->rows * out->columns; i++) out->data[i] = 0;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->columns; j++) {
            for (int k = 0; k < a->columns; k++) {
                out->data[i * out->columns + j] += (a->data[i * a->columns + k] * b->data[k * b->columns + j]);
            }
        }
    }
    return 1;
}

int transpose(Matrix *in, Matrix *out) {
    if (in->rows != out->columns || in->columns != out->rows) return 0;

    for (int i = 0; i < in->rows; i++) {
        for (int j = 0; j < in->columns; j++) {
            out->data[j * out->columns + i] = in->data[i * in->columns + j];
        }
    } 
    return 1;
}

int mv_mult(Matrix *a, Matrix *b, Matrix *out) {
    if (a->columns != b->rows) return 0;
    for (int i = 0; i < a->rows; i++) {
        out->data[i] = 0;
        for (int k = 0; k < a->columns; k++) {
            out->data[i] += a->data[i * a->columns + k] * b->data[k];
        }
    } 
    return 1;
}   

int element_wise_add(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->columns != b->columns) return 0;
    int r = a->rows;
    int c = a->columns;

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            a->data[i * c + j] += b->data[i * c + j];
        }
    }
    return 1;
}

int element_wise_sub(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->columns != b->columns) return 0;
    int r = a->rows;
    int c = a->columns;

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            a->data[i * c + j] -= b->data[i * c + j];
        }
    }
    return 1;
}

int element_wise_mult(Matrix *a, Matrix *b) {
    if (a->rows != b->rows || a->columns != b->columns) return 0;
    int r = a->rows;
    int c = a->columns;

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            a->data[i * c + j] *= b->data[i * c + j];
        }
    }
    return 1;
}

void scalar_mult(Matrix *a, double k) {
    int r = a->rows;
    int c = a->columns;

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            a->data[i * c + j] *= k;
        }
    }
}

float *vtof(Matrix *vector) {
    if (vector->rows > 1 && vector->columns > 1) {
        printf("Argument is not a vector.\n");
        return NULL;
    }
    u32 n = vector->rows * vector->columns;
    float *result = (float *)malloc(sizeof(float) * n);
    memcpy(result, vector->data, sizeof(float) * n);
    return result;
}
