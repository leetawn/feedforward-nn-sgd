#pragma once
#include "../include/config.h"

typedef struct Matrix {
    int rows, columns;
    float *data;
} Matrix;

Matrix *allocate_matrix(u32 rows, u32 columns);
void free_matrix(Matrix *in);
int multiply(Matrix *a, Matrix *b, Matrix *out);
int transpose(Matrix *in, Matrix *out);
int mv_mult(Matrix *a, Matrix *b, Matrix *out); 
int element_wise_add(Matrix *a, Matrix *b);
int element_wise_sub(Matrix *a, Matrix *b);
int element_wise_mult(Matrix *a, Matrix *b);
void scalar_mult(Matrix *a, double k);
float *vtof(Matrix *vector);