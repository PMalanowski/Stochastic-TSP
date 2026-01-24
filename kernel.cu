#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

// Makro do sprawdzania błędów CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Kernel inicjalizujący stany generatorów losowych (uruchamiany raz)
__global__ void initRNG(curandState* states, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &states[id]);
    }
}

// Główny kernel ewaluacji
// Każy wątek obsługuje JEDNEGO osobnika z populacji
__global__ void evaluatePopulationKernel(
    const int* __restrict__ populationPaths, // Spłaszczona tablica tras [POP_SIZE * NUM_CITIES]
    const float* __restrict__ means,         // Macierz średnich [NUM_CITIES * NUM_CITIES]
    const float* __restrict__ stddevs,       // Macierz odchyleń [NUM_CITIES * NUM_CITIES]
    float* __restrict__ fitnessResults,      // Tablica wynikowa [POP_SIZE]
    curandState* states,                     // Stany generatorów losowych
    int numCities,
    int popSize,
    int numSimulations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < popSize) {
        // Kopia stanu generatora do rejestru (szybciej)
        curandState localState = states[idx];

        // Wskaźnik na początek trasy tego konkretnego osobnika
        const int* myPath = &populationPaths[idx * numCities];

        // Zmienne do algorytmu Welforda lub prostej akumulacji momentów
        // Chcemy policzyć średnią i wariancję sumy czasów z K symulacji

        // Podejście uproszczone (ale wystarczające dla dużej liczby symulacji):
        // Liczymy sumę średnich i sumę wariancji trasy "w locie" dla każdej symulacji?
        // NIE. Wersja stochastyczna polega na tym, że losujemy wagi krawędzi.

        double sumTotalTime = 0.0;
        double sumSqTotalTime = 0.0;

        for (int sim = 0; sim < numSimulations; ++sim) {
            float currentRouteTime = 0.0f;

            for (int i = 0; i < numCities; ++i) {
                int u = myPath[i];
                int v = myPath[(i + 1) % numCities]; // Krawędź u -> v

                // Pobranie parametrów rozkładu krawędzi z pamięci globalnej
                // Używamy __ldg dla read-only cache (optymalizacja dla GPU compute capability >= 3.5)
                float mu = means[u * numCities + v];
                float sigma = stddevs[u * numCities + v];

                // Losowanie z rozkładu normalnego
                // curand_normal zwraca float z N(0,1)
                float edgeTime = mu + sigma * curand_normal(&localState);

                if (edgeTime < 0.001f) edgeTime = 0.001f; // Zabezpieczenie

                currentRouteTime += edgeTime;
            }

            sumTotalTime += currentRouteTime;
            sumSqTotalTime += (currentRouteTime * currentRouteTime);
        }

        // Zapisz zaktualizowany stan generatora
        states[idx] = localState;

        // Obliczenie statystyk z symulacji
        double mean = sumTotalTime / numSimulations;
        double variance = (sumSqTotalTime / numSimulations) - (mean * mean);
        double stdDev = sqrt(variance > 0 ? variance : 0.0);

        // Funkcja celu: Średnia + 2 * odchylenie (approx 97.7% confidence upper bound)
        // Można użyć 1.645 dla 95%
        fitnessResults[idx] = (float)(mean + 1.645 * stdDev);
    }
}

// Wrapper C++ wywoływany z GeneticAlgorithm
// Zarządza pamięcią GPU (alokuje tylko raz jeśli to możliwe, ale dla prostoty tu zrobimy alokacje per call lub użyjemy klasy pomocniczej)
extern "C" void launchEvaluation(
    const int* h_population,
    const float* d_means,   // Dane problemu już na GPU
    const float* d_stddevs, // Dane problemu już na GPU
    float* h_fitness,
    curandState* d_states,
    int numCities,
    int popSize,
    int numSimulations
) {
    // Alokacja pamięci dla populacji i wyników na GPU
    int* d_population;
    float* d_fitness;

    CUDA_CHECK(cudaMalloc((void**)&d_population, popSize * numCities * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_fitness, popSize * sizeof(float)));

    // Kopiowanie populacji z CPU (Host) do GPU (Device)
    // To jest narzut komunikacyjny (overhead)
    CUDA_CHECK(cudaMemcpy(d_population, h_population, popSize * numCities * sizeof(int), cudaMemcpyHostToDevice));

    // Konfiguracja siatki wątków
    int threadsPerBlock = 256;
    int blocksPerGrid = (popSize + threadsPerBlock - 1) / threadsPerBlock;

    evaluatePopulationKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_population,
        d_means,
        d_stddevs,
        d_fitness,
        d_states,
        numCities,
        popSize,
        numSimulations
    );

    // Czekaj na zakończenie
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kopiowanie wyników z powrotem na CPU
    CUDA_CHECK(cudaMemcpy(h_fitness, d_fitness, popSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Sprzątanie (tylko zmiennych tymczasowych per generacja)
    cudaFree(d_population);
    cudaFree(d_fitness);
}

// Funkcje pomocnicze do alokacji stałych danych (Problem Data)
extern "C" void allocateProblemOnGPU(const float* h_means, const float* h_stddevs, int numCities, float** d_means, float** d_stddevs) {
    size_t size = numCities * numCities * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)d_means, size));
    CUDA_CHECK(cudaMalloc((void**)d_stddevs, size));
    CUDA_CHECK(cudaMemcpy(*d_means, h_means, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_stddevs, h_stddevs, size, cudaMemcpyHostToDevice));
}

extern "C" void freeProblemOnGPU(float* d_means, float* d_stddevs) {
    cudaFree(d_means);
    cudaFree(d_stddevs);
}

extern "C" void initRNGStates(curandState** d_states, int popSize, unsigned long seed) {
    CUDA_CHECK(cudaMalloc((void**)d_states, popSize * sizeof(curandState)));
    int threads = 256;
    int blocks = (popSize + threads - 1) / threads;
    initRNG<<<blocks, threads>>>(*d_states, seed, popSize);
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void freeRNGStates(curandState* d_states) {
    cudaFree(d_states);
}