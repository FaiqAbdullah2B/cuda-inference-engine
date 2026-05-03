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
    int padded_vocab_size; // I don't understand the use of this yet. Let's see   
    int num_layers;
    int num_heads; // number of attention heads per block
    int channels;
    int num_parameters; // total trainable weights
} GPT2Config;

#define NUM_PARAMETER_TENSORS 2
typedef struct {
    float *wte; // weight token embeddings
    float *wpe; // weight positional encoding
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
    // float *cpu_losses;
} GPT2;

void init_parameters_sizes(size_t *param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C;            // wte
    param_sizes[1] = maxT * C;          // wpe
    // param_sizes[2] = L * C;             // ln1w
    // param_sizes[3] = L * C;             // ln1b
    // param_sizes[4] = L * (3 * C) * C;   // qkvw
    // param_sizes[5] = L * (3 * C);       // qkvb
    // param_sizes[6] = L * C * C;         // attprojw
    // param_sizes[7] = L * C;             // attprojb
}

void init_activation_sizes(size_t *act_sizes, int B, int T, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * C; // encoded
}

// kernels
__global__ void encoder_forward_kernel(float *out,
                                       int *inp, float *wte, float *wpe,
                                       int B, int T, int C) {
    int b = blockIdx.y * blockDim.y + threadIdx.y; // row
    int t = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (b >= 0 && b < B && t >= 0 && t < T) {
        // get the index of the token inp[b, t]
        int ix = inp[b * T + t];
        // get token embedding weight index
        float *wte_ix = wte + ix * C;
        // get token position weight index
        float *wpe_t = wpe + t * C;
        // add position encoding to token embeddings and store
        for (int i = 0; i < C; i++) {
            out[b * T * C + t * C + i] = wte_ix[i] + wpe_t[i];
        }
    }
}

// kernel launchers
void encoder_forward(float *out,
                     int *inp, float *wte, float *wpe,
                     int B, int T, int C) {
    const int blockSize_y = 16;
    const int blockSize_x = 32;
    const int gridSize_y = CEIL_DIV(B, blockSize_y);
    const int gridSize_x = CEIL_DIV(T, blockSize_x);
    dim3 gridDim = dim3(gridSize_x, gridSize_y, 1);
    dim3 blockDim = dim3(blockSize_x, blockSize_y, 1);
    encoder_forward_kernel<<<gridDim, blockDim>>>(out, inp, wte, wpe, B, T, C);
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
        &params->wte, &params->wpe
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
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < n; i++) {
        *(targets[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

float* malloc_and_point_activations(ActivationTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->encoded
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

    if (model->acts_memory == NULL) {
        model->batch_size = 0;
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
        // cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
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
    float *encoder_result = (float*)malloc(B * T * C * sizeof(float));
    cudaMemcpy(encoder_result, acts.encoded, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    printf("=================== encoder output ==================\n");
    for (int i = 0; i < B * T; i++) {
        printf("%f ", encoder_result[i]);
        if (i % 100 == 0) {
            printf("\n");
        }
    }
    printf("\n");

    free(encoder_result);
}

void gpt2_free(GPT2 *model) {
    cudaCheck(cudaFree(model->params_memory));
    cudaCheck(cudaFree(model->acts_memory));
    cudaCheck(cudaFree(model->inputs));
    cudaCheck(cudaFree(model->targets));
    // cudaFreeHost(model->cpu_losses);
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