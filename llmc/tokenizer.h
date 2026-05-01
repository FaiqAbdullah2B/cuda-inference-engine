#ifndef ENGINE_TOKENIZER_H
#define ENGINE TOKENIZER_H

#include "common.h"

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // last token
} Tokenizer;

void tokenizer_init(Tokenizer *t, const char *filename);
const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id);
void tokenizer_free(Tokenizer *t);

#endif
