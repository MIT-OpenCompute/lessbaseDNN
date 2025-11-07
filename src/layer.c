#include "trebuchet/layer.h"
#include <stdlib.h>

static Tensor* linear_forward(Layer *self, Tensor *input); 
static Tensor* activation_forward(Layer *self, Tensor *input);
static Tensor* loss_forward(Layer *self, Tensor *input);

static void linear_free(Layer *self);
static void activation_free(Layer *self);
static void loss_free(Layer *self);