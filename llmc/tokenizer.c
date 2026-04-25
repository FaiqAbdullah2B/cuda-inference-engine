#include "tokenizer.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

void tokenizer_init(Tokenizer *tokenizer, const char* filename) {
    FILE *file = fopen(filename, "rb");

    if (file == NULL) {
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        tokenizer->init_ok = 0;
        exit(EXIT_FAILURE);
    }
    
    // Read the header.
    uint32_t header[256];
    
    fread(header, sizeof(uint32_t), 256, file);
    
    //assert(header[0] == 2024038);
    
    int version = header[1]; // tokenizer version
    tokenizer->vocab_size = header[2]; // number of distinct tokens
    
    // Set EOF token number
    if (version == 1) { // It did not have EOF
        assert(tokenizer->vocab_size == 50257);
        tokenizer->eot_token = 50256;
    }
    else if (version == 2) {
        tokenizer->eot_token = header[3];
    }
    else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
    }

    unsigned char length;
    tokenizer->token_table = (char **)malloc(tokenizer->vocab_size * sizeof (char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        fread(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token must be as big  as 1 character
        char *token_bytes = (char *)malloc(length + 1);
        fread(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';
        tokenizer->token_table[i] = token_bytes;
    }

    fclose(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %u!\n", token_id);
        return NULL;
    }
}

void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}
