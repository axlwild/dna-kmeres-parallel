cmake_minimum_required(VERSION 3.14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

include_directories(SYSTEM ./src)
include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include)
include_directories(/usr/include/)

project(Examen CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Examen main.cu utils.h kernels.h)

set_target_properties(
        Examen
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
