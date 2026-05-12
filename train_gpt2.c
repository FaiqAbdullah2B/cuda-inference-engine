#include "./llmc/common.h"
#include "./llmc/dataloader.h"
#include "./llmc/tokenizer.h"
#include "./llmc/rand.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>  
#include <assert.h> 
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define HEADER_SIZE 256

typedef struct {
    int max_seq_len; // max sequence length
    int vocab_size; // fixed size
    int padded_vocab_size; // I don't understand the use of this yet. Let's see   
    int num_layers;
    int num_heads; // number of attention heads per block
    int channels;
    int num_parameters; // total trainable weights
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

#define NUM_ACTIVATION_TENSORS 15
typedef struct {
    float *encoded; // (B, T, C) // output of encoding
    float *ln1;     // (L, B, T, C) // output of first layer normalization
    float *qkv;     // (L, B, T, 3*C) // output of qkv
    float *atty;  // (L, B, T, C) // output of attention before projection
    float *preatt; // (L, B, NH, T, T) // pre attention cache to store QK
    float *att; // (L, B, NH, T, T) // attention cache to store softmax(QK)
    float *attproj; // (L, B, T, C) // attention output projected
    float *residual2; // (L, B, T, C)
    float *ln2; // (L, B, T, C)
    float *fch; // (L, B, T, 4*C)
    float *fch_gelu; // (L, B, T, 4*C)
    float *fcproj; // (L, B, T, C)
    float *residual3; // (L, B, T, C)
    float *lnf; // (B, T, C)
    float *logits; // (B, T, V)
} ActivationTensors;

typedef struct {
    double cpu_time;
    double allocation_time;
    int activations_memory;
} Benchmark;

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

void init_parameters_sizes(size_t *param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C;            // wte
    param_sizes[1] = maxT * C;          // wpe
    param_sizes[2] = L * C;             // ln1w
    param_sizes[3] = L * C;             // ln1b
    param_sizes[4] = L * (3 * C) * C;   // qkvw
    param_sizes[5] = L * (3 * C);       // qkvb
    param_sizes[6] = L * C * C;         // attprojw
    param_sizes[7] = L * C;             // attprojb
    param_sizes[8] = (size_t)L * C;           // ln2w
    param_sizes[9] = (size_t)L * C;           // ln2b
    param_sizes[10] = (size_t)L * 4 * C * C;  // fcw
    param_sizes[11] = (size_t)L * 4 * C;      // fcb
    param_sizes[12] = (size_t)L * C * 4 * C;  // fcprojw
    param_sizes[13] = (size_t)L * C;          // fcprojb
    param_sizes[14] = (size_t)C;              // lnfw
    param_sizes[15] = (size_t)C;              // lnfb
}

void init_benchmark(Benchmark *benchmark) {
    benchmark->allocation_time = 0.0f;
    benchmark->cpu_time = 0.0f;
    benchmark->activations_memory = 0;
}

// adds positional encoding and encoded token into an output
void encoder_forward(float *out,
                int *inp, float *wte, float *wpe,
                int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // find the output position in out[b, t, :]
            float *out_bt = out + b * T * C + t * C;
            // get the index of the token inp[b, t]
            int ix = inp[b * T + t];
            // get token embedding weight index
            float *wte_ix = wte + ix * C;
            // get token position weight index
            float *wpe_t = wpe + t * C;
            // add both to get 
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

// layer normalization
void layernorm_forward(float *out, float *inp, float *weight, float *bias,
                       int B, int T, int C) {
    float eps = 1e-5f; // a constant to prevent division by zero
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *x = inp + b * T * C + t * C;
            
            // mean calculation
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m /= C;

            // variance calculation
            float variance = 0.0f;
            for (int i = 0; i < C; i++) {
                float deviation = x[i] - m;
                variance += deviation * deviation;
            }
            variance /= C;

            // normalize and store
            float *out_ptr = out + b * T * C + t * C;
            float stdev = 1.0f / sqrtf((float)(variance + eps));
            for (int i = 0; i < C; i++) {
                float normalized = stdev * (x[i] - m);
                float output = normalized * weight[i] + bias[i];
                out_ptr[i] = output;
            }
        }
    }
}

void matmul_forward(float *out,
                    float *inp, float *weight, float *bias,
                    int B, int T, int C, int OC) {    
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *out_bt = out + b * T * OC + t * OC;
            float *inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float *wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

void attention_forward(float *out, float *preatt_cache, float *att_cache,
                       float *inp, int B, int T, int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0 / sqrtf(hs);
    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                // index to query
                float *queryt = inp + b * T * C3 + t * C3 + h * hs;
                float maxval = -FLT_MAX;
                // preceding sequence of words only
                for (int t2 = 0; t2 <= t; t2++) {
                    // index to key
                    float *keyt = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    
                    // perform val = (Q . K) / sqrt(hs)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += queryt[i] * keyt[i];
                    }
                    val *= scale;

                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_cache[t2] = val;
                }

                // softmax exponent calculation
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_cache[t2] - maxval);
                    expsum += expv;
                    att_cache[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                
                // normalization to get softmax
                for (int t2 = 0; t2 <= t; t2++) {
                    att_cache[t2] *= expsum_inv;
                }

                float *out_bth = out + b * T * C + t * C + h * hs;
                
                for (int i = 0; i < hs; i++) {
                    out_bth[i] = 0.0f;
                }

                // accumulate values from all the heads into out of attention
                for (int t2 = 0; t2 <= t; t2++) {
                    // value index
                    float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    // multiply and add
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_cache[t2] * value_t2[i];
                    }
                }
            }
        }
    }
}

void residual_forward(float *out, float *inp1, float *inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float *out, float *inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes) {
    size_t total_elements = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        total_elements += param_sizes[i];
    }
    
    // malloc all the parameters in one block of memory
    float *params_memory = (float*)malloc(total_elements * sizeof(float));
    if (params_memory == NULL) {
        printf("ERROR: Failed to allocate parameters memory\n");
        exit(1);
    }

    // casting the activation tensor pointer to a float** for easier assignment
    float **ptrs[NUM_PARAMETER_TENSORS] = {
        &params->wte, 
        &params->wpe, 
        &params->ln1w, 
        &params->ln1b, 
        &params->qkvw, 
        &params->qkvb, 
        &params->attprojw, 
        &params->attprojb, 
        &params->ln2w, 
        &params->ln2b, 
        &params->fcw, 
        &params->fcb, 
        &params->fcprojw, 
        &params->fcprojb, 
        &params->lnfw, 
        &params->lnfb
    };

    float *current_ptr = params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *ptrs[i] = current_ptr;
        current_ptr += param_sizes[i];
    }

    return params_memory;
}

float *malloc_and_point_activations(ActivationTensors *acts, size_t *act_sizes) {
    // Calculate the total size
    size_t total_elements = 0;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        total_elements += act_sizes[i];
    }
    
    float *acts_memory = (float*)malloc(total_elements * sizeof(float));
    if (acts_memory == NULL) {
        printf("ERROR: Failed to allocate activations memory\n");
        exit(1);
    }

    float **ptrs[NUM_ACTIVATION_TENSORS] = {
        &acts->encoded, 
        &acts->ln1, 
        &acts->qkv, 
        &acts->atty, 
        &acts->preatt, 
        &acts->att, 
        &acts->attproj, 
        &acts->residual2, 
        &acts->ln2, 
        &acts->fch, 
        &acts->fch_gelu, 
        &acts->fcproj, 
        &acts->residual3, 
        &acts->lnf, 
        &acts->logits
    };

    float *current_ptr = acts_memory;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *ptrs[i] = current_ptr;
        current_ptr += act_sizes[i];
    }

    return acts_memory;
}

void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint_path) {
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) {
        printf("Error: failed to open file %s\n", checkpoint_path);
        exit(1);
    }
    
    int model_header[HEADER_SIZE];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) {
        printf("Bad magic model file\n");
        exit(1);
    }

    size_t maxT, V, Vp, L, NH, C; // read all the sizes
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);
    
    // allocate space and read the parameters in
    init_parameters_sizes(model->param_sizes, model->config);
    
    // read the total size of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num parameters: %ld\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters now
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    fread(model->params_memory, sizeof(float), num_parameters, model_file);
    fclose(model_file);

    // other inits
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->seq_len = 0;
    model->batch_size = 0;
    model->mean_loss = -1.0f; // no loss
}

void gpt2_forward(GPT2 *model, int *inputs, int *targets, size_t B, size_t T,
                  Benchmark *benchmark) {
    if (model->params_memory == NULL) {
        printf("Error: uninitialized paramters\n");
        exit(1);
    }

    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    if (model->acts_memory == NULL) {
        double alloc_time_start = clock();

        model->batch_size = B;
        model->seq_len = T;

        // allocate space for activations
        model->act_sizes[0] = B * T * C;               // encoded
        model->act_sizes[1] = L * B * T * C;           // ln1
        model->act_sizes[2] = L * B * T * 3 * C;       // qkv
        model->act_sizes[3] = L * B * T * C;           // atty
        model->act_sizes[4] = L * B * NH * T * T;      // preatt
        model->act_sizes[5] = L * B * NH * T * T;      // att
        model->act_sizes[6] = L * B * T * C;           // attproj
        model->act_sizes[7] = L * B * T * C;           // residual2
        model->act_sizes[8] = L * B * T * C;           // ln2
        model->act_sizes[9] = L * B * T * 4 * C;       // fch
        model->act_sizes[10] = L * B * T * 4 * C;      // fch_gelu
        model->act_sizes[11] = L * B * T * C;          // fcproj
        model->act_sizes[12] = L * B * T * C;          // residual3
        model->act_sizes[13] = B * T * C;              // lnf
        model->act_sizes[14] = B * T * Vp;             // logits

        size_t num_activations = 0;
        for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }

        model->num_activations = num_activations;

        benchmark->activations_memory = (num_activations * sizeof(float)) >> 20;

        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        model->inputs = (int*)malloc(B * T * sizeof(int));
        model->targets = (int*)malloc(B * T * sizeof(int));

        double alloc_time_end = clock();

        benchmark->allocation_time = alloc_time_end - alloc_time_start;
    } 
    else {
        if (B != model->batch_size || T != model->seq_len) {
            printf("Error: Dynamic resizing not supported. Model: B=%d T=%d, Desired: B=%ld T=%ld\n",
                model->batch_size, model->seq_len, B, T);
            exit(1); // Abort to prevent buffer overflow
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    float* residual;

    // --- Embedding Layer ---
    
    double total_time_start = clock();

    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);

    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);

    double total_time_end = clock();

    benchmark->cpu_time += total_time_end - total_time_start;
}

// numerically stable softmax in-place over n elements
void softmax(float *x, int n) {
    float maxval = -FLT_MAX;
    for (int i = 0; i < n; i++) { if (x[i] > maxval) maxval = x[i]; }
    float expsum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxval); expsum += x[i]; }
    for (int i = 0; i < n; i++) { x[i] /= expsum; }
}

// sample one index from a probability distribution
// coin is a uniform random float in [0, 1)
int sample_multinomial(float *probs, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (coin < cdf) return i;
    }
    return n - 1; // fallback for floating point rounding
}

int main(int argc, char *argv[]) {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "../gpt2_124M.bin");

    const char* tiny_shake_train = "../dev/data/tinyshakespeare/tiny_shakespeare_train.bin";

    int B = 1;
    int T = 8;
    int batch_skips = 0;
    int max_sequence_len = 10;

    // optional command line args: B, T, batch_skips, max_sequence_len
    if (argc >= 2) {
        B = atoi(argv[1]);
    }
    if (argc >= 3) {
        T = atoi(argv[2]);
    }
    if (argc >= 4) {
        batch_skips = atoi(argv[3]);
    }
    if (argc >= 5) {
        max_sequence_len = atoi(argv[4]);
    }
    
    assert(B > 0 && T > 0 && batch_skips >= 0 && max_sequence_len > 0);

    Dataloader train_loader;
    dataloader_init(&train_loader, tiny_shake_train, B, T);

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "../gpt2_tokenizer.bin");

    srand(time(NULL));

    dataloader_next_batch(&train_loader);
    
    for (int i = 0; i < batch_skips; i++) {
        dataloader_next_batch(&train_loader);
    }

    int V  = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;

    // mutable context window seeded from the loaded batch
    int *gen_tokens = (int*)malloc(B * T * sizeof(int));
    memcpy(gen_tokens, train_loader.inputs, B * T * sizeof(int));

    float *probs = (float*)malloc(V * sizeof(float)); // reused every step

    Benchmark benchmark;
    init_benchmark(&benchmark);

    printf("--- PROMPT ---\n");
    for (int i = 0; i < B * T; i++) {
        printf("%s", tokenizer_decode(&tokenizer, train_loader.inputs[i])); 
    }

    // --- Generation loop ---
    printf("\n--- Generating ---\n");

    double time_start = clock();
    for (int step = 0; step < max_sequence_len; step++) {

        // forward pass — pass NULL targets (no loss needed during inference)
        gpt2_forward(&model, gen_tokens, NULL, B, T, &benchmark);

        // --- sample from batch item b=0, last position T-1 ---
        int b = 0;
        // logits layout: (B, T, Vp), so offset to [b, T-1, 0]
        float *logits_last = model.acts.logits + b * T * Vp + (T - 1) * Vp;

        // copy only the real vocab (ignore the padded tail) then softmax
        for (int i = 0; i < V; i++) probs[i] = logits_last[i];
        softmax(probs, V);

        float coin      = (float) rand() / (float) RAND_MAX;
        int next_token  = sample_multinomial(probs, V, coin);

        // decode and print
        if (tokenizer.init_ok) {
            const char *tok = tokenizer_decode(&tokenizer, next_token);
            printf("%s", tok);
        } else {
            printf("%d ", next_token);
        }
        fflush(stdout);

        // slide the context window left by 1, append the new token
        for (int t = 0; t < T - 1; t++) {
            gen_tokens[b * T + t] = gen_tokens[b * T + t + 1];
        }
        gen_tokens[b * T + (T - 1)] = next_token;
    }
    double time_end = clock();
    double time_taken = time_end - time_start;

    printf("\n\n--- BENCHMARK ---\n");
    printf("B = %d, T = %d, C = %d, Words Gen = %d\n", 
        B, T, model.config.channels, max_sequence_len);
    printf("Activations memory used: %d MB\n", benchmark.activations_memory);
    printf("Total execution time: %.2lfms\n", time_taken / 1000);
    printf("Allocation time overhead: %.2lfms\n", benchmark.allocation_time / 1000);
    printf("Total CPU execution time: %.2lfms\n", benchmark.cpu_time / 1000);
    printf("Average time per word generation: %.2lfms\n", 
        time_taken / (max_sequence_len * 1000));
    printf("Average time per GPU forward pass: %.2lfms\n", 
        benchmark.cpu_time / (max_sequence_len * 1000));

    printf("\n");

    free(gen_tokens);
    free(probs);

    dataloader_free(&train_loader);
    tokenizer_free(&tokenizer);
    free(model.params_memory);
    if (model.acts_memory != NULL) {
        free(model.acts_memory);
        free(model.inputs);
        free(model.targets);
    }
}
