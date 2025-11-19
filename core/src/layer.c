#include "../include/layer.h"
#include "../include/registry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ====================================================
// Layers
// ====================================================

static Layer* linear_create(LayerConfig *config);
static Layer* activation_create(LayerConfig *config);
static Tensor* linear_forward(Layer *self, Tensor *input);
static Tensor* relu_forward(Layer *self, Tensor *input);
static Tensor* sigmoid_forward(Layer *self, Tensor *input);
static Tensor* tanh_forward(Layer *self, Tensor *input);
static Tensor* softmax_forward(Layer *self, Tensor *input);

static Layer* linear_create(LayerConfig *config) {
    LinearParams *params = (LinearParams*)config->params;
    Layer *layer = malloc(sizeof(Layer));
    layer->name = strdup(config->name);
    layer->weights = tensor_randn((size_t[]){params->in_features, params->out_features}, 2, 42);
    
    float scale = sqrtf(2.0f / (float)params->in_features);
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->data[i] *= scale;
    }
    
    layer->bias = tensor_zeroes((size_t[]){params->out_features}, 1);
    layer->output = NULL;
    layer->parameters = malloc(2 * sizeof(Tensor*));
    layer->parameters[0] = layer->weights;
    layer->parameters[1] = layer->bias;
    layer->num_parameters = 2;
    layer->forward = linear_forward;
    
    layer->config_data_size = sizeof(LinearParams);
    layer->config_data = malloc(layer->config_data_size);
    memcpy(layer->config_data, params, layer->config_data_size);
    
    return layer;
}

static Layer* activation_create(LayerConfig *config) {
    Layer *layer = malloc(sizeof(Layer));
    layer->name = strdup(config->name);
    layer->weights = NULL;
    layer->bias = NULL;
    layer->output = NULL;
    layer->parameters = NULL;
    layer->num_parameters = 0;
    layer->forward = get_layer_forward_fn(config->name);
    
    layer->config_data = NULL;
    layer->config_data_size = 0;
    
    return layer;
}

static Tensor* linear_forward(Layer *self, Tensor *input) {
    if (!self || !input || !self->weights || !self->bias) return NULL;
    Tensor *Z_0 = tensor_matmul(input, self->weights);
    Tensor *Z = tensor_add(Z_0, self->bias);
    return Z;
}

static Tensor* relu_forward(Layer *self, Tensor *input) {
    return tensor_relu(input);
}

static Tensor* sigmoid_forward(Layer *self, Tensor *input) {
    return tensor_sigmoid(input);
}

static Tensor* tanh_forward(Layer *self, Tensor *input) {
    return tensor_tanh(input);
}

static Tensor* softmax_forward(Layer *self, Tensor *input) {
    return tensor_softmax(input);
}

// ====================================================
// Layer Registration
// ====================================================

void layer_register_builtins(void) {
    register_layer("linear", linear_create, linear_forward);
    register_layer("relu", activation_create, relu_forward);
    register_layer("sigmoid", activation_create, sigmoid_forward);
    register_layer("tanh", activation_create, tanh_forward);
    register_layer("softmax", activation_create, softmax_forward);
}

// ====================================================
// Layer Management
// ====================================================

Layer* layer_create(LayerConfig config) {
    LayerCreateFn create_fn = get_layer_create_fn(config.name);
    if (!create_fn) return NULL;
    return create_fn(&config);
}

void layer_free(Layer *layer) {
    if (!layer) return; 

    if (layer->name) free(layer->name);
    if (layer->weights) tensor_free(layer->weights);
    if (layer->bias) tensor_free(layer->bias);
    if (layer->output) tensor_free(layer->output);
    if (layer->parameters) free(layer->parameters);
    if (layer->config_data) free(layer->config_data);

    free(layer);
}

Tensor* layer_forward(Layer *layer, Tensor *input) {
    if (!layer || !layer->forward) return NULL; 
    return layer->forward(layer, input);
}

// ====================================================
// Autograd Utilities
// ====================================================

void layer_zero_grad(Layer *layer) {
    if (!layer) return;

    for (size_t i = 0; i < layer->num_parameters; i++) {
        if (layer->parameters[i]->grad) {
            tensor_zero_grad(layer->parameters[i]);
        }
    }
}

Tensor** layer_get_parameters(Layer *layer, size_t *num_params) {
    if (!layer || !num_params) return NULL;

    *num_params = layer->num_parameters;
    return layer->parameters;
}

