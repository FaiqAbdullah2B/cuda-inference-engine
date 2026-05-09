cmake_minimum_required(VERSION 3.22.1)

project("CUDA_INFERENCE_ENGINE" LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/bin/")

add_subdirectory(llmc)

add_executable(train_gpt2c train_gpt2.c)
add_executable(train_gpt2cu train_gpt2.cu)

target_link_libraries(train_gpt2c PRIVATE llmc m)
target_link_libraries(train_gpt2cu PRIVATE llmc m)