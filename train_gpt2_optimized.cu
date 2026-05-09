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

#define NUM_ACTIVATION_TENSORS 17
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
        // printf("allocated %ld MiB for activations\n", 
        //     (num_activations * sizeof(float)) >> 20);
        
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

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "../gpt2_tokenizer.bin");

    int B = 1;          
    int T = 128;
    
    const char* tiny_shake_train = "../dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    Dataloader train_loader;
    dataloader_init(&train_loader, tiny_shake_train, B, T);
    
    // Fetch a batch to use as prompt
    dataloader_next_batch(&train_loader);
    dataloader_next_batch(&train_loader);
    dataloader_next_batch(&train_loader);
    dataloader_next_batch(&train_loader);

    printf("--- PROMPT ---\n");
    for (int i = 0; i < B * T; i++) {
        printf("%s", tokenizer_decode(&tokenizer, train_loader.inputs[i])); 
    }
    printf("\n--- GENERATION ---\n");

    // Seed the random number generator
    srand(time(NULL));

    int max_new_tokens = 100;

    generate_text(&model, &tokenizer, train_loader.inputs, B, T, max_new_tokens);

    printf("\n------------------\n");

    dataloader_free(&train_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    
    return 0;
}