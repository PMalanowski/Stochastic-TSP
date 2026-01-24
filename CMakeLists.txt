cmake_minimum_required(VERSION 3.20)
project(StochasticTSP CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# --- OPCJA KONFIGURACYJNA ---
# Domyślnie włączona (ON). Na laptopie wyłączysz ją ręcznie.
option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Definiujemy makro, które będzie widoczne w kodzie C++
    add_compile_definitions(USE_CUDA)

    # Dodajemy kernel tylko jeśli CUDA jest włączona
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
else()
    # Wersja bez CUDA (tylko cpp)
    message(STATUS "Building WITHOUT CUDA support")
    add_executable(StochasticTSP main.cpp) # Bez kernel.cu!

    target_link_libraries(StochasticTSP
            PRIVATE
            MPI::MPI_CXX
            OpenMP::OpenMP_CXX
    )
endif()