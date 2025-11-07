#include <stdio.h>
#include <stdlib.h>
#include "trebuchet/trebuchet.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define NUM_SAMPLES 50
#define NUM_EPOCHS 5
#define LEARNING_RATE 0.01f

void generate_data(Tensor *X, Tensor *y) {
    for (size_t i = 0; i < NUM_SAMPLES; i++) {
        for (size_t j = 0; j < INPUT_SIZE; j++) {
            X->data[i * INPUT_SIZE + j] = ((float)rand() / RAND_MAX) * 0.1f;
        }
        int label = rand() % OUTPUT_SIZE;
        for (size_t j = 0; j < OUTPUT_SIZE; j++) {
            y->data[i * OUTPUT_SIZE + j] = (j == label) ? 1.0f : 0.0f;
        }
    }
}

int main() {
    srand(42);
    
    printf("MNIST\n\n");
    
    size_t data_shape[] = {NUM_SAMPLES, INPUT_SIZE};
    size_t label_shape[] = {NUM_SAMPLES, OUTPUT_SIZE};
    Tensor *X = tensor_create(data_shape, 2);
    Tensor *y = tensor_create(label_shape, 2);
    generate_data(X, y);
    
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(INPUT_SIZE, HIDDEN_SIZE)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(HIDDEN_SIZE, OUTPUT_SIZE)));
    network_add_layer(net, layer_create(SOFTMAX()));
    network_add_layer(net, layer_create(CE_LOSS(y)));
    
    Optimizer *opt = optimizer_create(net, SGD(LEARNING_RATE, 0.9f));

    printf("Training...\n");
    network_train(net, opt, X, y, NUM_EPOCHS, NUM_SAMPLES, 1);
    network_remove_last_layer(net);
    
    printf("\nInference...\n");
    size_t single_shape[] = {1, INPUT_SIZE};
    Tensor *input = tensor_create(single_shape, 2);
    for (size_t i = 0; i < INPUT_SIZE; i++) {
        input->data[i] = X->data[i];
    }
    
    Tensor *pred = network_forward(net, input);
    int pred_class = 0;
    for (size_t i = 1; i < OUTPUT_SIZE; i++) {
        if (pred->data[i] > pred->data[pred_class]) pred_class = i;
    }
    
    int true_class = 0;
    for (size_t i = 0; i < OUTPUT_SIZE; i++) {
        if (y->data[i] == 1.0f) { true_class = i; break; }
    }
    
    printf("Sample 0 - True: %d, Predicted: %d\n", true_class, pred_class);
    
    tensor_free(input);
    tensor_free(X);
    tensor_free(y);
    optimizer_free(opt);
    network_free(net);
    
    return 0;
}
