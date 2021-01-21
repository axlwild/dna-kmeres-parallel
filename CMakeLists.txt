cmake_minimum_required(VERSION 3.14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")


project(Examen CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Examen main.cu)

set_target_properties(
        Examen
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
