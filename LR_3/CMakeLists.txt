cmake_minimum_required(VERSION 3.25)
project(LR_3 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(LR_3 main.cu)

set_target_properties(LR_3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
