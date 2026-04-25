#ifndef ENGINE_DATALOADER_H
#define ENGINE_DATALOADER_H

#include "common.h"

typedef struct {
    size_t B;                   // number of examples in a batch
    size_t T;                   // tokens per example
    FILE *tokens_file;
    long file_size;
    long current_position;
    uint16_t *buffer;
    // size_t num_batches;
    // int *inputs;
    // int *targets;
} Dataloader;

int dataloader_init(Dataloader *loader, const char *filename, int B, int T);
void dataloader_next_batch(Dataloader *loader, int *inputs, int *targets);
void dataloader_reset(Dataloader *loader);
void dataloader_free(Dataloader *loader);

#endif
