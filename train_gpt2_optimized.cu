#include "./llmc/common.h"
extern "C" {
    #include "./llmc/dataloader.h"
    #include "./llmc/tokenizer.h"
}
#include "./llmc/rand.h"
#include "./dev/cuda/utils.h"
#include <math.h>
#include <mma.h>

using namespace nvcuda;

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

#define NUM_HALF_TENSORS 2
typedef struct {
    half *qkvi;
    half *qkvw;
} HalfTensors;

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
    // halfs
    HalfTensors halfs;
    size_t half_sizes[NUM_HALF_TENSORS];
    half *halfs_memory;
    size_t num_halfs;

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

void init_half_sizes(size_t *half_sizes, int B, int T, GPT2Config config) {
    size_t L = config.num_layers;
    size_t C = config.channels;
    half_sizes[0] = L * B * T * C;      // layernorm 1 (qkvi)
    half_sizes[1] = L * (3 * C) * C;    // qkvw
}

// kernels
#define ENCODER_BLOCK_DIM_Y 16
__global__ void encoder_forward_kernel(float *out,
                                       int *inp, float *wte, float *wpe,
                                       int B, int T, int C) {
    int b = blockIdx.z * blockDim.z + threadIdx.z; // depth
    int t = blockIdx.y * blockDim.y + threadIdx.y; // row
    int c = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // col

    // shared memory because every single c will access the same token id
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

__global__ void float_to_half_kernel(half* out, const float* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]); // Hardware conversion
    }
}

// UNDERSTAND THIS or revert this
#define TS 16
// M x K * K x N Matmul
__global__ void matmul_forward_kernel(float *out,
                                      half *inp, half *weight, float *bias,
                                      int B, int T, int C, int OC) {
    int M = B * T;
    int K = C;
    int N = OC;

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // exact thread within a block 0 - 127

    __shared__ half inp_s[TS * 2][TS];
    __shared__ half weight_s[TS][TS * 2];

    int block_row_start = blockIdx.y * 32;
    int block_col_start = blockIdx.x * 32;

    if (block_row_start >= M || block_col_start >= N) return;

    wmma::fragment<wmma::matrix_a, TS, TS, TS, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TS, TS, TS, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TS, TS, TS, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // input is M x K
    for (int k = 0; k < K; k += TS) {
        // load into shared memory.
        for (int i = 0; i < 4; i++) {
            int load_idx = tid + (i * 128);
            int r = load_idx / 16;
            int c = load_idx % 16;

            int global_row_a = block_row_start + r;
            int global_col_a = k + c;

            if (global_row_a < M && global_col_a < K) {
                inp_s[r][c] = inp[global_row_a * K + global_col_a];
            } else {
                inp_s[r][c] = 0.0f; // Pad with zero
            }
        }

        // weight is N x K not K x N, so each column is K elements long
        for (int i = 0; i < 4; i++) {
            int load_idx = tid + (i * 128);
            int r = load_idx / 32; // K dimension
            int c = load_idx % 32; // N dimension

            int global_k = k + c; 
            // block col start is quite confusing for this.
            int global_n = block_col_start + r; // starting N + r

            if (global_k < K && global_n < N) {
                weight_s[r][c] = weight[global_n * K + global_k];
            } else {
                weight_s[r][c] = 0.0f; // Pad with zero
            }
        }
        __syncthreads();

        // point to the start
        half *a_ptr = &inp_s[(threadIdx.y / 2) * 16][0];
        half *b_ptr = &weight_s[0][(threadIdx.y % 2) * 16];

        wmma::load_matrix_sync(a_frag, a_ptr, 16);
        wmma::load_matrix_sync(b_frag, b_ptr, 32);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    int warp_row_local = (threadIdx.y / 2) * 16;
    int warp_col_local = (threadIdx.y % 2) * 16;

    int global_warp_row = block_row_start + warp_row_local;
    int global_warp_col = block_col_start + warp_col_local;

    if (global_warp_row < M && global_warp_col < N) {
        float *c_ptr = out + (global_warp_row * N) + global_warp_col;
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);

        for (int i = 0; i < TS*TS; i+=32) {
            int element_idx = i + threadIdx.x;
            int r = element_idx / TS;
            int c = element_idx % TS;

            int cur_row = global_warp_row + r;
            int cur_col = global_warp_col + c;

            if (cur_row < M && cur_col < N) {
                c_ptr[r * N + c] += bias[cur_col];
            }
        }
    }
}

__global__ void residual_forward_kernel(float *out, float *inp1, float *inp2, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= 0 && i <= N - 4) {
        float4 inp1_val = *(float4 *)(&inp1[i]);
        float4 inp2_val = *(float4 *)(&inp2[i]);

        inp1_val.x = inp1_val.x + inp2_val.x;
        inp1_val.y = inp1_val.y + inp2_val.y;
        inp1_val.z = inp1_val.z + inp2_val.z;
        inp1_val.w = inp1_val.w + inp2_val.w;
        
        *(float4 *)(&out[i]) = inp1_val;
    }

    // if N is not divisble by 4
    if (i == N - (N % 4)) {
        for (int j = 0; j < (N % 4); ++j) {
            out[i + j] = inp1[i + j] + inp2[i + j];
        }
    }
}

#define M_PI 3.14159265358979323846 // math.h won't help...
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel(float *out, float *inp, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= 0 && i <= N - 4) {
        float4 inp_val = *(float4 *)(&inp[i]);
        float4 cube;
        cube.x = 0.044715f * inp_val.x * inp_val.x * inp_val.x;
        cube.y = 0.044715f * inp_val.y * inp_val.y * inp_val.y;
        cube.z = 0.044715f * inp_val.z * inp_val.z * inp_val.z;
        cube.w = 0.044715f * inp_val.w * inp_val.w * inp_val.w;

        cube.x = 0.5f * inp_val.x * (1.0f + tanhf(GELU_SCALING_FACTOR * (inp_val.x + cube.x)));
        cube.y = 0.5f * inp_val.y * (1.0f + tanhf(GELU_SCALING_FACTOR * (inp_val.y + cube.y)));
        cube.z = 0.5f * inp_val.z * (1.0f + tanhf(GELU_SCALING_FACTOR * (inp_val.z + cube.z)));
        cube.w = 0.5f * inp_val.w * (1.0f + tanhf(GELU_SCALING_FACTOR * (inp_val.w + cube.w)));

        *(float4 *)(&out[i]) = cube;
    }

    // if N is not divisble by 4
    if (i == N - (N % 4)) {
        for (int j = 0; j < (N % 4); ++j) {
            float x = inp[i + j];
            float cube = 0.044715f * x * x * x;
            cube = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
            out[i + j] = cube;
        }
    }
}

__device__ float reduction_max(float val, float *input_s) {
    int tx = threadIdx.x;

    input_s[tx] = val;
    __syncthreads();

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            // hardware accelerated instruction for max. Intrinsic.
            input_s[tx] = fmaxf(input_s[tx], input_s[tx + stride]);
        }
        __syncthreads();
    }

    return input_s[0];
}

// probs <- softmax(logits)
#define TEMPERATURE 0.75f
__global__ void softmax_forward_kernel(float *probs, float *logits, 
                                       int B, int T, int V, int Vp) {
    int row_idx = blockIdx.x; // One block processes one row

    if (row_idx >= B * T) return;
    
    // shared mem to perform reduction max on
    __shared__ float max_s[1024];

    // max specific to this thread
    float thread_max = -1e20f;

    // Calculate Max
    // 1024 threads * 4 = 4096 elements processed per loop iteration
    for (int i = threadIdx.x * 4; i < V; i += blockDim.x * 4) {
        if (i <= Vp - 4) {
            float4 logits_val = *(float4 *)&logits[row_idx * Vp + i];
            // maxval is only calculated and subtracted for numerical stability
            logits_val.x /= TEMPERATURE;
            logits_val.y /= TEMPERATURE;
            logits_val.z /= TEMPERATURE;
            logits_val.w /= TEMPERATURE;

            float local_max = fmax(logits_val.x, logits_val.y);
            local_max = fmax(local_max, logits_val.z);
            local_max = fmax(local_max, logits_val.w);

            thread_max = fmaxf(local_max, thread_max);
        }
        else {
            for (int j = 0; j < V - i; j++) {
                float val = logits[row_idx * Vp + i + j] / TEMPERATURE;
                thread_max = fmaxf(thread_max, val);
            }

        }
    }

    float maxval = reduction_max(thread_max, max_s);

    // Sum the exps for the denominator
    float thread_sum = 0.0f;
    for (int i = threadIdx.x * 4; i < V; i += blockDim.x * 4) {
        if (i <= V - 4) {
            // pray that its in cache and go for another global mem access
            float4 logits_val = *(float4 *)&logits[row_idx * Vp + i];

            logits_val.x /= TEMPERATURE;
            logits_val.y /= TEMPERATURE;
            logits_val.z /= TEMPERATURE;
            logits_val.w /= TEMPERATURE;

            float local_sum = 0.0f;
            local_sum += expf(logits_val.x - maxval);
            local_sum += expf(logits_val.y - maxval);
            local_sum += expf(logits_val.z - maxval);
            local_sum += expf(logits_val.w - maxval);

            thread_sum += local_sum;
        }
        else {
            for (int j = 0; j < V - i; ++j) {
                float val = logits[row_idx * Vp + i + j] / TEMPERATURE;
                thread_sum += expf(val - maxval);
            }
        }
    }

    // recycling max shared mem to save resources
    float sum = reduction_sum(thread_sum, max_s);

    // Normalize and store
    for (int i = threadIdx.x * 4; i < V; i += blockDim.x * 4) {
        if (i <= V - 4) {
            float4 val = *(float4 *)&logits[row_idx * Vp + i];

            val.x /= TEMPERATURE;
            val.y /= TEMPERATURE;
            val.z /= TEMPERATURE;
            val.w /= TEMPERATURE;

            // Calculate the final probability for all 4 floats
            float4 probs_val;
            probs_val.x = expf(val.x - maxval) / sum;
            probs_val.y = expf(val.y - maxval) / sum;
            probs_val.z = expf(val.z - maxval) / sum;
            probs_val.w = expf(val.w - maxval) / sum;

            // Vectorized Write to Global Memory
            *(float4 *)&probs[row_idx * Vp + i] = probs_val;
        }
        else {
            // Scalar cleanup for the last 1, 2, or 3 elements
            for (int j = 0; j < V - i; ++j) {
                float val = logits[row_idx * Vp + i + j] / TEMPERATURE;
                probs[row_idx * Vp + i + j] = expf(val - maxval) / sum;
            }
        }
    }

    // Padding
    for (int i = V + threadIdx.x; i < Vp; i += blockDim.x) {
        probs[row_idx * Vp + i] = 0.0f;
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
    int gridDim_x = CEIL_DIV(C, blockDim_x * 4);
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

void matmul_forward(float *out, half* qkvi, half* qkvw,
                    float *inp, float *weight, float *bias,
                    int B, int T, int C, int OC) {
    dim3 gridDim_inp(CEIL_DIV(B * T * C, 512), 1, 1);
    dim3 blockDim_fth(512, 1, 1);
    float_to_half_kernel<<<gridDim_inp, blockDim_fth>>>(qkvi, inp, B * T * C);
    cudaCheck(cudaGetLastError());

    int weight_elements = OC * C;
    dim3 gridDim_wt(CEIL_DIV(weight_elements, 512), 1, 1);
    float_to_half_kernel<<<gridDim_wt, blockDim_fth>>>(qkvw, weight, weight_elements);
    cudaCheck(cudaGetLastError());


    int M = B * T;
    // K is C
    int N = OC;

    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32), 1);
    dim3 blockDim(32, 4, 1); // 4 warps per block

    matmul_forward_kernel<<<gridDim, blockDim>>>(
        out, qkvi, qkvw, bias, B, T, C, OC);
    cudaCheck(cudaGetLastError());

}

void residual_forward(float *out, float *inp1, float *inp2, int N) {
    int blockDim_x = 512;
    int gridDim_x = CEIL_DIV(N, blockDim_x * 4);
    dim3 gridDim(gridDim_x, 1, 1);
    dim3 blockDim(blockDim_x, 1, 1);
    residual_forward_kernel<<<gridDim, blockDim>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward(float *out, float *inp, int N) {
    int blockDim_x = 512;
    int gridDim_x = CEIL_DIV(N, blockDim_x * 4);
    dim3 gridDim(gridDim_x, 1, 1);
    dim3 blockDim(blockDim_x, 1, 1);
    gelu_forward_kernel<<<gridDim, blockDim>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void softmax_forward(float* probs, float* logits, 
                    int B, int T, int V, int Vp) {
    // Vp is 50304
    int blockDim_x = 1024;
    int gridDim_x = B * T; // one block handles the entire width
    dim3 gridDim(gridDim_x, 1, 1);
    dim3 blockDim(blockDim_x, 1, 1);
    softmax_forward_kernel<<<gridDim, blockDim>>>(probs, logits, B, T, V, Vp);
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

half* malloc_and_point_halfs(HalfTensors* halfs, const size_t* half_sizes) {
    size_t num_halfs = 0;
    for (size_t i = 0; i < NUM_HALF_TENSORS; i++) {
        num_halfs += half_sizes[i];
    }

    half *halfs_memory;
    cudaCheck(
        cudaMalloc((void **)&halfs_memory, num_halfs * sizeof(half))
    );

    half **ptrs[] = {
        &halfs->qkvi, &halfs->qkvw
    };

    half *halfs_memory_iterator = halfs_memory;
    for (size_t i = 0; i < NUM_HALF_TENSORS; i++) {
        *(ptrs[i]) = halfs_memory_iterator;
        halfs_memory_iterator += half_sizes[i];
    }
    return halfs_memory;
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

    if (model->halfs_memory != NULL) {
        if (B != model->batch_size || T != model->seq_len) {
            cudaFree(model->halfs_memory);
            model->halfs_memory = NULL;
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

    if (model->halfs_memory == NULL) {
        init_half_sizes(model->half_sizes, B, T, model->config);

        size_t num_halfs = 0;
        for (size_t i = 0; i < NUM_HALF_TENSORS; i++) {
            num_halfs += model->half_sizes[i];
        }

        model->num_halfs = num_halfs;
        model->halfs_memory = malloc_and_point_halfs(
            &model->halfs, model->half_sizes);
        printf("allocated %ld MiB for halfs\n", 
            (num_halfs * sizeof(half)) >> 20);
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
    HalfTensors halfs = model->halfs;
    float *residual;
    float *intermediate = (float *)malloc(B * T * 4 * C * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Encoder time taken: %f ms\n", milliseconds);

    cudaEventRecord(start);
    layernorm_forward(acts.ln1, acts.encoded, params.ln1w, params.ln1b, B, T, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0.0f;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Layernorm time taken: %f ms\n", milliseconds);

    cudaEventRecord(start);
    matmul_forward(acts.qkv, halfs.qkvi, halfs.qkvw, acts.ln1, params.qkvw, params.qkvb, B, T, C, 3 * C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0.0f;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("QKV time taken: %f ms\n", milliseconds);

    cudaMemcpy(intermediate, acts.qkv, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=================== QKV output ==================\n");
    for (int i = 0; i < B * T; i++) {
        printf("%f ", intermediate[i]);
        if ((i + 1) % 100 == 0) {
            printf("\n");
        }
    }
    printf("\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
    int B = 24;
    int T = 256;
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