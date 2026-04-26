#include "dataloader.h"

int dataloader_init(Dataloader *loader, const char *filename, int B, int T) {
    loader->B = B;
    loader->T = T;
    
    // reminder: revisit this for shards
    loader->tokens_file = fopen(filename, "rb");

    if (!loader->tokens_file) {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        exit(1);
    }

    uint32_t header[HEADER_SIZE];
    fread(header, sizeof(uint32_t), HEADER_SIZE, loader->tokens_file);

    if (header[0] != 20240520) {
        printf("Bad magic in the data file\n");
        exit(EXIT_FAILURE);
    }
    if (header[1] != 1) {
        printf("Bad version in the data file\n");
    }
    long ntok = header[2]; // number of tokens in the file
    assert(ntok > 0);
    
    fseek(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseek(loader->tokens_file, 0, SEEK_SET);

    long expected_file_size = HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
    if (loader->file_size != expected_file_size) {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }

    loader->buffer = (uint16_t*)malloc((B * T + 1) * sizeof(uint16_t));
    loader->inputs = (int*)malloc(B * T * sizeof(int));
    loader->targets = (int*)malloc(B * T * sizeof(int));

    dataloader_reset(loader);

    fseek(loader->tokens_file, HEADER_SIZE * sizeof(int), SEEK_SET);
    return ntok;
}

void dataloader_next_batch(Dataloader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    
    size_t batch_size = B*T;
    int header_bytes = HEADER_SIZE * sizeof(int);
    
    fseek(loader->tokens_file, header_bytes + loader->current_position * sizeof(uint16_t), SEEK_SET);
    fread(loader->buffer, sizeof(uint16_t), batch_size + 1, loader->tokens_file);

    for (int i = 0; i < batch_size; i++) {
        loader->inputs[i] = loader->buffer[i];
        loader->targets[i] = loader->buffer[i + 1];
    }
    loader->current_position += batch_size;
    
    // Reminder: revisit wrap around logic later for shards.
    if ((loader->current_position + batch_size + 1) * sizeof(uint16_t) >= loader->file_size) {
        loader->current_position = 0; // wrap around
    }
}

void dataloader_reset(Dataloader *loader) {
    long header_bytes = HEADER_SIZE * sizeof(int);
    fseek(loader->tokens_file, header_bytes, SEEK_SET);

    loader->current_position = 0;
}

void dataloader_free(Dataloader *loader) {
    if (loader->tokens_file) fclose(loader->tokens_file);
    if (loader->inputs) free(loader->inputs);
    if (loader->targets) free(loader->targets);
}
