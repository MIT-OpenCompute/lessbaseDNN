#include "optimizer.h"
#include "registry.h"
#include <stdlib.h>
#include <string.h> 
#include <math.h>

// ====================================================
// Optimizer States
// ====================================================

typedef struct {
    float learning_rate;
    float momentum;
    Tensor **velocity;
} SGDState;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    Tensor **m;
    Tensor **v;
} AdamState;

static void* sgd_init_state(Tensor **parameters, size_t num_parameters, void *params);
static void sgd_step(Optimizer *opt);
static void sgd_free_state(void *state, size_t num_parameters);

static void* adam_init_state(Tensor **parameters, size_t num_parameters, void *params);
static void adam_step(Optimizer *opt);
static void adam_free_state(void *state, size_t num_parameters);

// ====================================================
// SGD
// ====================================================

static void* sgd_init_state(Tensor **parameters, size_t num_parameters, void *params) {
    SGDParams *p = (SGDParams*)params;
    SGDState *state = malloc(sizeof(SGDState));
    state->learning_rate = p->learning_rate;
    state->momentum = p->momentum;
    state->velocity = NULL;
    
    if (state->momentum > 0.0f) {
        state->velocity = malloc(num_parameters * sizeof(Tensor*));
        for (size_t i = 0; i < num_parameters; i++) {
            state->velocity[i] = tensor_create(parameters[i]->shape, parameters[i]->ndim);
            tensor_fill(state->velocity[i], 0.0f);
        }
    }
    return state;
}

static void sgd_step(Optimizer *opt) {
    SGDState *state = (SGDState*)opt->state;
    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i];
        if (!param->grad) continue;
        
        if (state->momentum > 0.0f) {
            for (size_t j = 0; j < param->size; j++) {
                state->velocity[i]->data[j] = state->momentum * state->velocity[i]->data[j] 
                                             - state->learning_rate * param->grad[j];
                param->data[j] += state->velocity[i]->data[j];
            }
        } else {
            for (size_t j = 0; j < param->size; j++) {
                param->data[j] -= state->learning_rate * param->grad[j];
            }
        }
    }
}

static void sgd_free_state(void *state, size_t num_parameters) {
    SGDState *s = (SGDState*)state;
    if (s->velocity) {
        for (size_t i = 0; i < num_parameters; i++) {
            tensor_free(s->velocity[i]);
        }
        free(s->velocity);
    }
    free(s);
}

// ====================================================
// ADAM
// ====================================================

static void* adam_init_state(Tensor **parameters, size_t num_parameters, void *params) {
    AdamParams *p = (AdamParams*)params;
    AdamState *state = malloc(sizeof(AdamState));
    state->learning_rate = p->learning_rate;
    state->beta1 = p->beta1;
    state->beta2 = p->beta2;
    state->epsilon = p->epsilon;
    state->t = 0;
    
    state->m = malloc(num_parameters * sizeof(Tensor*));
    state->v = malloc(num_parameters * sizeof(Tensor*));
    
    for (size_t i = 0; i < num_parameters; i++) {
        state->m[i] = tensor_create(parameters[i]->shape, parameters[i]->ndim);
        state->v[i] = tensor_create(parameters[i]->shape, parameters[i]->ndim);
        tensor_fill(state->m[i], 0.0f);
        tensor_fill(state->v[i], 0.0f);
    }
    return state;
}

static void adam_step(Optimizer *opt) {
    AdamState *state = (AdamState*)opt->state;
    state->t += 1;
    
    float bias_correction1 = 1.0f - powf(state->beta1, state->t);
    float bias_correction2 = 1.0f - powf(state->beta2, state->t);
    
    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i];
        if (!param->grad) continue;
        
        for (size_t j = 0; j < param->size; j++) {
            state->m[i]->data[j] = state->beta1 * state->m[i]->data[j] + (1.0f - state->beta1) * param->grad[j];
            state->v[i]->data[j] = state->beta2 * state->v[i]->data[j] + (1.0f - state->beta2) * param->grad[j] * param->grad[j];
            
            float m_hat = state->m[i]->data[j] / bias_correction1;
            float v_hat = state->v[i]->data[j] / bias_correction2;
            param->data[j] -= state->learning_rate * m_hat / (sqrtf(v_hat) + state->epsilon);
        }
    }
}

static void adam_free_state(void *state, size_t num_parameters) {
    AdamState *s = (AdamState*)state;
    for (size_t i = 0; i < num_parameters; i++) {
        tensor_free(s->m[i]);
        tensor_free(s->v[i]);
    }
    free(s->m);
    free(s->v);
    free(s);
}

// ====================================================
// Optimizer Registration
// ====================================================

void optimizer_register_builtins(void) {
    register_optimizer("sgd", sgd_init_state, sgd_step, sgd_free_state);
    register_optimizer("adam", adam_init_state, adam_step, adam_free_state);
}

// ====================================================
// Optimizer Operations
// ====================================================

Optimizer* optimizer_create(Tensor **parameters, size_t num_parameters, OptimizerConfig config) {
    if (!parameters || num_parameters == 0) return NULL;

    OptimizerInitStateFn init_fn = get_optimizer_init_state_fn(config.name);
    OptimizerStepFn step_fn = get_optimizer_step_fn(config.name);
    OptimizerFreeStateFn free_fn = get_optimizer_free_state_fn(config.name);
    
    if (!init_fn || !step_fn || !free_fn) return NULL;

    Optimizer *opt = malloc(sizeof(Optimizer));
    if (!opt) return NULL;

    opt->name = strdup(config.name);
    opt->parameters = parameters;
    opt->num_parameters = num_parameters;
    opt->step = step_fn;
    opt->zero_grad = optimizer_zero_grad;
    opt->free_state = free_fn;
    opt->state = init_fn(parameters, num_parameters, config.params);

    if (!opt->state) {
        free(opt->name);
        free(opt);
        return NULL;
    }

    return opt;
}



void optimizer_step(Optimizer *opt) {
    if (!opt || !opt->step) return;
    opt->step(opt);
}

void optimizer_zero_grad(Optimizer *opt) {
    if (!opt) return; 

    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i]; 
        if (param->grad) {
            tensor_zero_grad(param);
        }
    }
}

void optimizer_free(Optimizer *opt) {
    if (!opt) return;

    if (opt->state && opt->free_state) {
        opt->free_state(opt->state, opt->num_parameters);
    }
    
    if (opt->name) free(opt->name);
    free(opt);
}