#include "./llmc/common.h"
#include <math.h>

typedef struct {
    int max_seq_len; // max sequence length
    int vocab_size; // fixed size
    int padded_vocab_size; // I don't understand the use of this yet. Let's see   
    int num_layers;
    int num_heads; // number of attention heads per block
    int channels;
    int num_parameters; // total trainable weights
} GPT2Config;

#define NUM_PARAMETER_TENSORS 2
typedef struct {
    float *wte; // weight token embeddings
    float *wtp; // weight token positioning
    // float *ln1w; // 1st layer normalization's weights
    // float *ln1b;
    // float *qkvw;
    // float *qkvb;
    // float *attprojw;
    // float *attprojb;
} ParameterTensors;

#define NUM_ACTIVATION_TENSORS 1
typedef struct {
    float *encoded; // output of encoding
    // float *ln1;     // output of first layer normalization
    // float *qkv;     // output of qkv
    // float *atty;
    // float *preatt;  // pre attention cache to store QK
    // float *att;
} ActivationTensors;

typedef struct {
    GPT2Config config;
    // parameters
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;
    // activations
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float *acts_memory;
    size_t num_activations;

    int batch_size; // B
    int seq_len; // T
    int *inputs;
    int *targets;
    float mean_loss;
} GPT2;

int main() {

}