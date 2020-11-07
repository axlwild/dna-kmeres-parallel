cmake_minimum_required(VERSION 3.14)
project(Examen CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Examen main.cu)
add_executable(Examen_Prueba main.cpp)

set_target_properties(
        Examen
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)