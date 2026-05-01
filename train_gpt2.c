#include "./llmc/common.h"
#include "./llmc/dataloader.h"
#include "./llmc/tokenizer.h"
#include "./llmc/rand.h"
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

#define NUM_PARAMETER_TENSORS 4
typedef struct {
    float *wte; // weight token embeddings
    float *wtp; // weight token positioning
    float *ln1w; // 1st layer normalization's weights
    float *ln1b;
//    float *ln2w;
//    float *ln2b;
//    float *lnfw;
//    float *lnfb;
} ParameterTensors;

typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;
    
    int batch_size; // B
    int seq_len; // T
    int *inputs;
    int *targets;
    float mean_loss;
} GPT2;

void init_parameters_sizes(size_t *param_sizes, GPT2Config config) {
    size_t Vp = config.vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C;       // wte
    param_sizes[1] = maxT * C;     // wpe
    param_sizes[2] = L * C;        // ln1w
    param_sizes[3] = L * C;        // ln1b
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

float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes) {
    size_t num_params = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_params += param_sizes[i];
    }
    
    // malloc all the parameters in one block of memory
    float *params_memory = (float*)malloc(num_params * sizeof(float));

    // assign all the tensors
    float **ptrs[] = {
        &params->wte, &params->wtp, &params->ln1w, &params->ln1b
    };
    
    // set all the starting locations for pointers
    float *params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
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
        printf("Bag magic model file\n");
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
    model->inputs = NULL;
    model->targets = NULL;
    model->seq_len = 0;
    model->batch_size = 0;
    model->mean_loss = -1.0f; // no loss
}

void gpt2_forward(GPT2 *model, int *inputs, int *targets, size_t B, size_t T) {
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

    ParameterTensors params = model->params;
    float *output_encoded = (float*)malloc(sizeof(float) * B * T * C);
    encoder_forward(output_encoded, inputs, params.wte, params.wtp, B, T, C);
    
    printf("=================== encoder output ==================\n");
    for (int i = 0; i < B * T; i++) {
        printf("%f ", output_encoded[i]);
    }
    printf("\n"); 
    float *ln1w = params.ln1w;
    float *ln1b = params.ln1b;
    float *ln1 = (float*)malloc(sizeof(float) * B * T * C);
    layernorm_forward(ln1, output_encoded, ln1w, ln1b, B, T, C);

    printf("=================== layernorm output ==================\n");
    for (int i = 0; i < B * T; i++) {
        printf("%f ", ln1[i]);
    }

    printf("\n");
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
}
