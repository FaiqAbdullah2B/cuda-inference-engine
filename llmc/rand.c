#include "rand.h"

void manual_seed(uint64_t seed) {
    srand(seed);
}

static float random_f32() {
    return (float)rand() / (float)RAND_MAX;
}

float random_normal() {
    float u1 = random_f32();
    float u2 = random_f32();
    
    // Ensure u1 is not zero to avoid log(0)
    if (u1 < 1e-7f) u1 = 1e-7f;

    // The Box-Muller formula
    // sqrt(-2 * log(u1)) * cos(2 * pi * u2)
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}
