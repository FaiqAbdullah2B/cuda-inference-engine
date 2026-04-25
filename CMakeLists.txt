cmake_minimum_required(VERSION 3.28.3)

project("CUDA_INFERENCE_ENGINE")

set(C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/bin/")

add_subdirectory(llmc)
add_executable(main main.c)
target_link_libraries(main PRIVATE llmc)
