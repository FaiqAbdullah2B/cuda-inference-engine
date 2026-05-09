#include "./llmc/common.h"
extern "C" {
    #include "./llmc/dataloader.h"
    #include "./llmc/tokenizer.h"
}
#include "./llmc/rand.h"
#include "./dev/cuda/utils.h"
#include <math.h>

typedef struct {
    int max_seq_len; // max sequence length
    int vocab_size; // fixed size
    int padded_vocab_size;
    int num_layers;
    int num_heads; // number of attention heads per block
    int channels;
    int num_parameters; // total trainable weights
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float *wte;         // weight token embeddings
    float *wpe;         // weight positional encoding
    float *ln1w;        // 1st normalization's weights
    float *ln1b;        // 1st normalization's biases
    float *qkvw;        // QKV weights
    float *qkvb;        // QKV biases
    float *attprojw;    // Attention weights
    float *attprojb;    // Attention biases
    float *ln2w;        // 2nd normalization's weights
    float *ln2b;        // 2nd normalization's biases
    float *fcw;         // Fully connected weights
    float *fcb;         // Fully connected biases
    float *fcprojw;     // Gelu weights
    float *fcprojb;     // Gelu biases
    float *lnfw;        // final normalization's weights
    float *lnfb;        // final normalization's biases
} ParameterTensors;

#define NUM_ACTIVATION_TENSORS 16
typedef struct {
    float *encoded; // output of encoding
    float *ln1;     // output of first layer normalization
    float *qkv;     // output of qkv
    float *atty;
    float *preatt;  // pre attention cache to store QK
    float *att;
    float *attproj;
    float *residual2;
    float *ln2; // (L, B, T, C)
    float *fch; // (L, B, T, 4*C)
    float *fch_gelu; // (L, B, T, 4*C)
    float *fcproj; // (L, B, T, C)
    float *residual3; // (L, B, T, C)
    float *lnf; // (B, T, C)
    float *logits; // (B, T, V)
    float *probs; // (B, T, V)
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
    param_sizes[8] = L * C;             // ln2w
    param_sizes[9] = L * C;             // ln2b
    param_sizes[10] = L * (4 * C) * C;  // fcw
    param_sizes[11] = L * (4 * C);      // fcb
    param_sizes[12] = L * C * (4 * C);  // fcprojw
    param_sizes[13] = L * C;            // fcprojb
    param_sizes[14] = C;                // lnfw
    param_sizes[15] = C;                // lnfb
}

void init_activation_sizes(size_t *act_sizes, int B, int T, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // layernorm 1
    act_sizes[2] = L * B * T * C * 3; // qkv
    act_sizes[3] = L * B * T * C;        // atty
    act_sizes[4] = L * B * NH * T * T;   // preatt
    act_sizes[5] = L * B * NH * T * T;   // att
    act_sizes[6] = L * B * T * C; // attproj
    act_sizes[7] = L * B * T * C; // residual2
    act_sizes[8] = L * B * T * C; // ln2
    act_sizes[9] = L * B * T * 4*C; // fch
    act_sizes[10] = L * B * T * 4*C; // fch_gelu
    act_sizes[11] = L * B * T * C; // fcproj
    act_sizes[12] = L * B * T * C; // residual3
    act_sizes[13] = B * T * C; // lnf
    act_sizes[14] = B * T * Vp; // logits
    act_sizes[15] = B * T * Vp; // probs
}

// kernels
#define ENCODER_BLOCK_DIM_Y 16
__global__ void encoder_forward_kernel(float *out,
                                       int *inp, float *wte, float *wpe,
                                       int B, int T, int C) {
    int b = blockIdx.z * blockDim.z + threadIdx.z; // depth
    int t = blockIdx.y * blockDim.y + threadIdx.y; // row
    int c = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // col

    __shared__ int token_ids[ENCODER_BLOCK_DIM_Y];

    if (b >= 0 && b < B && t >= 0 && t < T && c >= 0 && c < C) {
        if (threadIdx.x == 0) {
            // Store the token in the specific slot for this row
            token_ids[threadIdx.y] = inp[b * T + t];
        }
    }
    __syncthreads();
    
    if (b >= 0 && b < B && t >= 0 && t < T && c >= 0 && c < C) {
        // using float4 to reduce 12 accesses per 4 values to just 3 accesses
        float4 *wte_vec = (float4*)(&wte[token_ids[threadIdx.y] * C + c]);
        float4 *wpe_vec = (float4*)(&wpe[t * C + c]);

        float4 wte_val = *wte_vec;
        float4 wpe_val = *wpe_vec;
        float4 out_val;
        
        // add position encoding to token embeddings
        out_val.x = wte_val.x + wpe_val.x; // first val
        out_val.y = wte_val.y + wpe_val.y;
        out_val.z = wte_val.z + wpe_val.z;
        out_val.w = wte_val.w + wpe_val.w; 

        // store
        *((float4 *)(&out[b * T * C + t * C + c])) = out_val;
    }
}

#define EPSILON 1e-5f
#define LAYERNORM_BLOCK_DIM_Y 4
#define LAYERNORM_BLOCK_DIM_X 256 // hard coded to 256 for reduction sum.

__device__ float reduction_sum(float val, float* input_s) {
    int tx = threadIdx.x;

    input_s[tx] = val;
    __syncthreads();

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input_s[tx] += input_s[tx + stride];
        }        
        __syncthreads();
    }

    return input_s[0];
}

__global__ void layernorm_forward_kernel(float *out, float *inp, 
                                         const float * __restrict__ weight,
                                         const float * __restrict__ bias, 
                                         int B, int T, int C) {
    int b = blockIdx.z * blockDim.z + threadIdx.z; // depth
    int t = blockIdx.y * blockDim.y + threadIdx.y; // row
    int c = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // col
    
    __shared__ float mean_s[LAYERNORM_BLOCK_DIM_Y][256];

    float4 x;
    float4 weights;
    float4 biases;
    float mean = 0.0f;
    if (b >= 0 && b < B && t >= 0 && t < T && c >= 0 && c < C) {
        // index to the token
        x = *(float4 *)&inp[b * T * C + t * C + c];
        // add local values
        mean += (x.x + x.w + x.y + x.z);
    }

    mean = reduction_sum(mean, mean_s[threadIdx.y]);
    mean /= C;

    float var = 0.0f;
    float4 deviation;
    if (b < B && t < T && c < C) {
        deviation.x = x.x - mean;
        deviation.y = x.y - mean;
        deviation.z = x.z - mean;
        deviation.w = x.w - mean;
        var = (deviation.x * deviation.x) + (deviation.y * deviation.y) +
                    (deviation.z * deviation.z) + (deviation.w * deviation.w);
    }

    var = reduction_sum(var, mean_s[threadIdx.y]);
    var /= C;

    if (b >= 0 && b < B && t >= 0 && t < T && c >= 0 && c < C) {
        // normalize and store
        float stdev = 1.0f / sqrtf((float)(var + EPSILON));
        weights = *(float4 *)&weight[c];
        biases = *(float4 *)&bias[c];

        // reuse deviation to save space in register :')
        deviation.x = deviation.x * stdev * weights.x + biases.x;
        deviation.y = deviation.y * stdev * weights.y + biases.y;
        deviation.z = deviation.z * stdev * weights.z + biases.z;
        deviation.w = deviation.w * stdev * weights.w + biases.w;
        
        *(float4 *)(&out[b * T * C + t * C + c]) = deviation;
    }
}

// kernel launchers
void encoder_forward(float *out,
                    int *inp, float *wte, float *wpe,
                    int B, int T, int C) {
    int blockDim_z = 1;
    int blockDim_y = ENCODER_BLOCK_DIM_Y;
    int blockDim_x = 32;
    int gridDim_z = CEIL_DIV(B, blockDim_z);
    int gridDim_y = CEIL_DIV(T, blockDim_y);
    int gridDim_x = CEIL_DIV(C / 4, blockDim_x);
    dim3 blockDim = dim3(blockDim_x, blockDim_y, blockDim_z);
    dim3 gridDim = dim3(gridDim_x, gridDim_y, gridDim_z);
    encoder_forward_kernel<<<gridDim, blockDim>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward(float *out, float *inp, float *weight,
                       float* bias, int B, int T, int C) {
    int blockDim_z = 1;
    int blockDim_y = LAYERNORM_BLOCK_DIM_Y;
    int blockDim_x = LAYERNORM_BLOCK_DIM_X;

    int gridDim_z = CEIL_DIV(B, blockDim_z);
    int gridDim_y = CEIL_DIV(T, blockDim_y);
    int gridDim_x = 1; // one block handles the entire width

    dim3 blockDim = dim3(blockDim_x, blockDim_y, blockDim_z);
    dim3 gridDim = dim3(gridDim_x, gridDim_y, gridDim_z);
    layernorm_forward_kernel<<<gridDim, blockDim>>>(
        out, inp, weight, bias, B, T, C);
    cudaCheck(cudaGetLastError());
}

float *malloc_and_point_parameters(ParameterTensors* params, 
                                   size_t *param_sizes, bool on_device) {
    // on_device CPU = false and GPU = true

    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }

    float *params_memory;
    if (on_device) {
        cudaCheck(
            cudaMalloc((void **)&params_memory, num_parameters * sizeof(float))
        );
    }
    else {
        params_memory = (float *)malloc(num_parameters * sizeof(float));
    }

    float **ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, 
        &params->qkvw, &params->qkvb, &params->attprojw, &params->attprojb,
        &params->ln2w, &params->ln2b, &params->fcw, &params->fcb, 
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };

    float *params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *ptrs[i] = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

float* malloc_and_point(float** targets[], const size_t* act_sizes, int n) {
    size_t num_activations = 0;
    for (size_t i = 0; i < n; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, num_activations * sizeof(float)));
    cudaCheck(cudaMemset(acts_memory, 0, num_activations * sizeof(float))); // init with 0
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < n; i++) {
        *(targets[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

float* malloc_and_point_activations(ActivationTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->qkv, &acts->atty, &acts->preatt, 
        &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->fch, 
        &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf, 
        &acts->logits, &acts->probs
    };
    return malloc_and_point(ptrs, act_sizes, NUM_ACTIVATION_TENSORS);
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {
    // read in model from a checkpoint file
    FILE *model_file = fopen(checkpoint_path, "rb");
    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { fprintf(stderr, "Bad magic model file\n"); exit(EXIT_FAILURE); }
    if (model_header[1] != 3) {
        fprintf(stderr, "Bad version in model file\n");
        exit(EXIT_FAILURE);
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate space for all the parameters and read them in
    init_parameters_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

    // read in all the parameters from file and copy them to device
    float* params_memory_cpu = (float*)malloc(num_parameters * sizeof(float));
    fread(params_memory_cpu, sizeof(float), num_parameters, model_file);
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    free(params_memory_cpu);
    fclose(model_file);

    // other inits
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 *model, int *inputs, int *targets, int B, int T) {
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    if (model->acts_memory != NULL) {
        if (B != model->batch_size || T != model->seq_len) {
            cudaFree(model->acts_memory);
            cudaFree(model->inputs);
            cudaFree(model->targets);
            model->acts_memory = NULL;
        }
    }

    if (model->acts_memory == NULL) {
        model->batch_size = B;
        model->seq_len = T;

        init_activation_sizes(model->act_sizes, B, T, model->config);

        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }

        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(
            &model->acts, model->act_sizes);
        printf("allocated %ld MiB for activations\n", 
            (num_activations * sizeof(float)) >> 20);
        
        // also create memory for caching inputs and targets
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
    }
    else {
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    cudaCheck(
        cudaMemcpy(model->inputs, inputs, 
                   B * T * sizeof(int), 
                   cudaMemcpyHostToDevice)
    );
    if (targets != NULL) {
        cudaCheck(
            cudaMemcpy(model->targets, targets, 
                       B * T * sizeof(int), 
                       cudaMemcpyHostToDevice)
        );
    }

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    float *residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);
    
    layernorm_forward(acts.ln1, acts.encoded, params.ln1w, params.ln1b, B, T, C);
}

void generate_text(GPT2 *model, Tokenizer *tokenizer, int *prompt_tokens, int B, int T, int max_new_tokens) {
    int current_len = B * T;

    // Allocate a buffer large enough for the prompt + generated tokens
    int total_capacity = current_len + max_new_tokens;
    int *sequence = (int*)malloc(total_capacity * sizeof(int));
    memcpy(sequence, prompt_tokens, B * T * sizeof(int));

    float *h_probs = (float*)malloc(model->config.vocab_size * sizeof(float));

    for (int i = 0; i < max_new_tokens; i++) {
        gpt2_forward(model, sequence, NULL, 1, current_len);

        // get the softmax probabilities for the generated sequence
        float *d_last_probs = model->acts.probs + (current_len - 1) * model->config.padded_vocab_size;

        // copy the probabilities to host
        cudaMemcpy(h_probs, d_last_probs, 
                   model->config.vocab_size * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Multinomial sampling
        float coin_flip = (float)rand() / (float)RAND_MAX;
        float cdf = 0.0f;
        int next_token = model->config.vocab_size - 1; // Safe fallback
        
        for (int v = 0; v < model->config.vocab_size; v++) {
            cdf += h_probs[v];
            if (coin_flip < cdf) {
                next_token = v;
                break;
            }
        }

        // append the picked token and increment current_len by 1
        sequence[current_len] = next_token;
        current_len++;

        printf("%s", tokenizer_decode(tokenizer, next_token));
    }
    
    printf("\n");
    free(h_probs);
    free(sequence);
}

void gpt2_free(GPT2 *model) {
    cudaCheck(cudaFree(model->params_memory));
    cudaCheck(cudaFree(model->acts_memory));
    cudaCheck(cudaFree(model->inputs));
    cudaCheck(cudaFree(model->targets));
}

int main() {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "../gpt2_124M.bin");

    const char* tiny_shake_train = "../dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    int B = 4;
    int T = 64;
    Dataloader train_loader;
    dataloader_init(&train_loader, tiny_shake_train, B, T);

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "../gpt2_tokenizer.bin");

    dataloader_next_batch(&train_loader);
    gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);

    dataloader_free(&train_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
}