#ifndef REGISTRY_H
#define REGISTRY_H

#include "tensor.h"

struct Layer;
struct LayerConfig;
struct Network;
struct Optimizer;
struct OptimizerConfig;

// ====================================================
// Layer Registers
// ====================================================

typedef struct Layer* (*LayerCreateFn)(struct LayerConfig *config);
typedef Tensor* (*LayerForwardFn)(struct Layer *self, Tensor *input);

void register_layer(const char *name, LayerCreateFn create_fn, LayerForwardFn forward_fn);
LayerCreateFn get_layer_create_fn(const char *name);
LayerForwardFn get_layer_forward_fn(const char *name);

// ====================================================
// Tensor Operation Registers
// ====================================================

typedef Tensor* (*OpFn)(Tensor *a, Tensor *b);

void register_operation(const char *name, OpFn op_fn);
void register_operation_backend(const char *name, OpFn op_fn, int priority);
OpFn get_operation_fn(const char *name);

#define register_loss(name, fn) register_operation(name, fn)
#define get_loss_fn(name) get_operation_fn(name)
typedef OpFn LossFn;

typedef void (*BackwardFn)(Tensor *output);

void register_tensor_op(const char *name, BackwardFn backward_fn);
BackwardFn get_tensor_op_backward_fn(const char *name);

// ====================================================
// Optimizer Registers
// ====================================================

typedef void* (*OptimizerInitStateFn)(Tensor **parameters, size_t num_parameters, void *params);
typedef void (*OptimizerStepFn)(struct Optimizer *opt);
typedef void (*OptimizerFreeStateFn)(void *state, size_t num_parameters);

void register_optimizer(const char *name, 
                       OptimizerInitStateFn init_state_fn,
                       OptimizerStepFn step_fn,
                       OptimizerFreeStateFn free_state_fn);
OptimizerInitStateFn get_optimizer_init_state_fn(const char *name);
OptimizerStepFn get_optimizer_step_fn(const char *name);
OptimizerFreeStateFn get_optimizer_free_state_fn(const char *name);

// ====================================================
// Registry Initialization
// ====================================================

void registry_init();
void registry_cleanup();

#endif
