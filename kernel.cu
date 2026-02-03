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
// Każy wątek obsługuje jednego osobnika z populacji - analogiczne do Monte Carlo w Individual.h
__global__ void evaluatePopulationKernel(
    const int* __restrict__ populationPaths,
    const float* __restrict__ means,
    const float* __restrict__ stddevs,
    float* __restrict__ fitnessResults,
    curandState* states,
    int numCities,
    int popSize,
    int numSimulations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < popSize) {
        curandState localState = states[idx];


        const int* myPath = &populationPaths[idx * numCities];
        double sumTotalTime = 0.0;
        double sumSqTotalTime = 0.0;

        for (int sim = 0; sim < numSimulations; ++sim) {
            float currentRouteTime = 0.0f;

            for (int i = 0; i < numCities; ++i) {
                int u = myPath[i];
                int v = myPath[(i + 1) % numCities];
                float mu = means[u * numCities + v];
                float sigma = stddevs[u * numCities + v];

                float edgeTime = mu + sigma * curand_normal(&localState);

                if (edgeTime < 0.001f) edgeTime = 0.001f;

                currentRouteTime += edgeTime;
            }

            sumTotalTime += currentRouteTime;
            sumSqTotalTime += (currentRouteTime * currentRouteTime);
        }

        states[idx] = localState;

        double mean = sumTotalTime / numSimulations;
        double variance = (sumSqTotalTime / numSimulations) - (mean * mean);
        double stdDev = sqrt(variance > 0 ? variance : 0.0);

        // zeby nie zapisywac wynikow i nie sortowac to taki wzor dziala
        fitnessResults[idx] = (float)(mean + 1.645 * stdDev);
    }
}


extern "C" void launchEvaluation(
    const int* h_population,
    const float* d_means,
    const float* d_stddevs,
    float* h_fitness,
    curandState* d_states,
    int numCities,
    int popSize,
    int numSimulations
) {
    int* d_population;
    float* d_fitness;

    CUDA_CHECK(cudaMalloc((void**)&d_population, popSize * numCities * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_fitness, popSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_population, h_population, popSize * numCities * sizeof(int), cudaMemcpyHostToDevice));

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

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_fitness, d_fitness, popSize * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_population);
    cudaFree(d_fitness);
}

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