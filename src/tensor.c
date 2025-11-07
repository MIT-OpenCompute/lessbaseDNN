#include "tensor.h"
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <math.h> 

// Autograd forward declarations of backwards
static void backward_add(Tensor *out); 
static void backward_sub(Tensor *out);
static void backward_mul(Tensor *out);
static void backward_matmul(Tensor *out);
static void backward_transpose(Tensor *out);


// Autograd helper functions
static void topological_sort_util(Tensor *T, Tensor **visited, size_t *visited_count, Tensor **stack, size_t *stack_count, size_t max_size) {
    for (size_t i = 0; i < *visited_count; i++) {
        if (visited[i] == T) return; 
    }

    visited[(*visited_count)++] = T; 

    if (T->inputs) {
        for (size_t i = 0; i < T->num_inputs; i++) {
            if (T->inputs[i] && T->inputs[i]->requires_grad) {
                topological_sort_util(T->inputs[i], visited, visited_count, stack, stack_count, max_size); 
            }
        }
    }

    if (*stack_count < max_size) {
        stack[(*stack_count)++] = T;
    }
}

// Tensor creation/destruction
Tensor* tensor_create(size_t *shape, size_t ndim) {
    Tensor *T = (Tensor *)malloc(sizeof(Tensor)); 
    if (!T) return NULL; 

    T->ndim = ndim; 
    T->shape = (size_t *)malloc(ndim * sizeof(size_t)); 
    if (!T->shape) {
        free(T);
        return NULL;
    }

    T->size = 1; 
    for (size_t i = 0; i < ndim; i++) {
        T->shape[i] = shape[i]; 
        T->size *= shape[i]; 
    }

    T->data = (float *)malloc(T->size * sizeof(float)); 
    if (!T->data) {
        free(T->shape);
        free(T);
        return NULL;
    }

    T->grad = NULL; 
    T->requires_grad = 0;
    T->op = OP_NONE;
    T->inputs = NULL;
    T->num_inputs = 0;
    T->backward_fn = NULL;
    T->extra_data = NULL;
}

Tensor* tensor_zeroes(size_t *shape, size_t ndim) {
    Tensor *T = tensor_create(shape, ndim); 
    if (!T) return NULL; 

    memset(T->data, 0, T->size * sizeof(float)); 
    return T; 
}

Tensor* tensor_ones(size_t *shape, size_t ndim) {
    Tensor *T = tensor_create(shape, ndim); 
    if (!T) return NULL; 

    for (size_t i = 0; i < T->size; i++) {
        T->data[i] = 1.0f; 
    }
    return T;
}

Tensor* tensor_randn(size_t *shape, size_t ndim, int seed) {
    Tensor *T = tensor_create(shape, ndim); 
    if (!T) return NULL; 

    srand(seed); 
    for (size_t i = 0; i < T->size; i++) {
        float u1 = ((float)rand() / RAND_MAX);
        float u2 = ((float)rand() / RAND_MAX);
        T->data[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2); 
    }
    return T; 
}

void tensor_free(Tensor *T) {
    if (!T) return; 
    if (T->data) free(T->data); 
    if (T->grad) free(T->grad); 
    if (T->shape) free(T->shape); 
    if (T->inputs) free(T->inputs); 
    free(T); 
}

// Autograd functions
void tensor_set_requires_grad(Tensor *T, int requires_grad) {
    if (T) {
        T->requires_grad = requires_grad; 
    }
}

void tensor_backward(Tensor *T) {
    
}