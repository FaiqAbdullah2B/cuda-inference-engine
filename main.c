#include <stdio.h>

#include "./llmc/tokenizer.h"

int main(int argc, char *argv[]) {
    Tokenizer t;
    tokenizer_init(&t, "../gpt2_tokenizer.bin");

    const char* buf[512];
    int tokens[5] = {1212, 318, 257, 1332, 13};
    int ntokens = 5;
    
    for (int i = 0; i < ntokens; i++) {
        buf[i] = tokenizer_decode(&t, tokens[i]);
    }
    
    buf[ntokens] = "\0";

    for (int i = 0; i < ntokens; i++) {
        printf("%s", buf[i]);
    }
    
    printf("\n");
    tokenizer_free(&t);
}
