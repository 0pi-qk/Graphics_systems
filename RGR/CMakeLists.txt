cmake_minimum_required(VERSION 3.25)
project(RGR CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(RGR main.cu)

set_target_properties(RGR PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
