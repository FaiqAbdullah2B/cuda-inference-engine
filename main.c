#include <stdio.h>

#include "./llmc/tokenizer.h"
#include "./llmc/dataloader.h"

int main(int argc, char *argv[]) { 
    // loading data
    Dataloader l;
    int ntok = dataloader_init(&l, "../dev/data/tinyshakespeare/tiny_shakespeare_train.bin", 4, 64);
    printf("%d\n", ntok);

    const int B = 4;
    const int T = 64;
    const int batch_size = B*T;

    int inputs[batch_size];
    int targets[batch_size];
    dataloader_next_batch(&l, inputs, targets);

    dataloader_free(&l);
    
    // tokenizing data
    Tokenizer t;
    tokenizer_init(&t, "../gpt2_tokenizer.bin");
    
    const char* buf[batch_size];
    const char* targetBuf[batch_size];
    //int tokens[3] = {464, 3290, 318};
    //int ntokens = 3;
    
    for (int i = 0; i < batch_size; i++) {
        buf[i] = tokenizer_decode(&t, inputs[i]);
        targetBuf[i] = tokenizer_decode(&t, targets[i]);
    }

    for (int i = 0; i < batch_size; i++) {
        printf("%s", buf[i]);
    }
    printf("\n\n--------------------------- Targets ---------------------------\n\n");
    for (int i = 0; i < batch_size; i++) {
        printf("%s", targetBuf[i]);
    }
    printf("\n");
    tokenizer_free(&t);

    return 0;
}
