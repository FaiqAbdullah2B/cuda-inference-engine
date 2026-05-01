add_library(
    llmc dataloader.c tokenizer.c
)

target_include_directories(llmc PUBLIC .)
