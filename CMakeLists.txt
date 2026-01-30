cmake_minimum_required(VERSION 3.20)

# KROK 1: Usuwamy CUDA z definicji projektu. Zostawiamy tylko CXX (C++)
project(StochasticTSP CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# KROK 2: Definiujemy opcję (Domyślnie ON - dla Twojego PC)
option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA)
    # KROK 3: Tutaj dynamicznie włączamy język CUDA
    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(USE_CUDA)

    # Dodajemy plik .cu do kompilacji
    add_executable(StochasticTSP main.cpp kernel.cu)

    target_link_libraries(StochasticTSP
            PRIVATE
            MPI::MPI_CXX
            OpenMP::OpenMP_CXX
            CUDA::cudart
    )

    set_target_properties(StochasticTSP PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "75"
    )

    message(STATUS "CUDA support ENABLED")
else()
    # Wersja bez CUDA
    add_executable(StochasticTSP main.cpp
            mainSeq.cpp) # Bez kernel.cu!

    target_link_libraries(StochasticTSP
            PRIVATE
            MPI::MPI_CXX
            OpenMP::OpenMP_CXX
    )

    message(STATUS "CUDA support DISABLED (CPU Only build)")
endif()

add_executable(StochasticTSP_Seq mainSeq.cpp)