cmake_minimum_required(VERSION 3.20)
project(StochasticTSP CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
# 1. ZNAJDŹ OPENMP
find_package(OpenMP REQUIRED)

add_executable(StochasticTSP main.cpp kernel.cu)

target_link_libraries(StochasticTSP
        PRIVATE
        MPI::MPI_CXX
        CUDA::cudart
        OpenMP::OpenMP_CXX # 2. LINKUJ OPENMP
)

set_target_properties(StochasticTSP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75"
)