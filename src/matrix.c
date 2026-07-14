#include "../include/matrix.h"
#include <stdlib.h>
#include <stdio.h>


void allocate_matrix(Matrix *in, int rows, int columns) {
    in = (Matrix *)malloc(sizeof(in));
    in->rows = rows;
    in->columns = columns;
    in->data = (double *)calloc((rows * columns), sizeof(double));
}

void free_matrix(Matrix *in) {
    free(in);
}

int multiply(Matrix *a, Matrix *b, Matrix *out) {
    if (a->columns != b->rows) return 0;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->columns; j++) {
            for (int k = 0; k < a->columns; k++) {
                out->data[i * out->columns + j] += (a->data[i * a->columns + k] * b->data[k * b->columns + j]);
            }
        }
    }
    return 1;
}

void transpose(Matrix *in) {
    int r = in->rows;
    int c = in->columns;

    for (int j = 0; j < c; j++) {
        for (int i = 0; i < r; i++) {
            in->data[i * c + j] = in->data[j * r + i];
        }
    }
}

int mv_mult(Matrix *a, Matrix *b, Matrix *out) {
    if (a->columns != b->columns) return 0;
    for (int i = 0; i < a->rows; i++) {
        for (int k = 0; k < a->columns; k++) {
            out->data[i] += a->data[i * a->columns + k] * b->data[k];
        }
    } 
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

void scalar_mult(Matrix *a, int k) {
    int r = a->rows;
    int c = a->columns;

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            a->data[i * c + j] *= k;
        }
    }
}
