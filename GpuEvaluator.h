#ifndef STOCHASTICTSP_GPUEVALUATOR_H
#define STOCHASTICTSP_GPUEVALUATOR_H

#ifdef USE_CUDA

#include <vector>
#include "TSPProblem.h"
#include "Individual.h"

// Deklaracje funkcji zewnętrznych ("C" linkage) z kernel.cu
extern "C" void launchEvaluation(const int* h_population, const float* d_means, const float* d_stddevs, float* h_fitness, void* d_states, int numCities, int popSize, int numSimulations);
extern "C" void allocateProblemOnGPU(const float* h_means, const float* h_stddevs, int numCities, float** d_means, float** d_stddevs);
extern "C" void freeProblemOnGPU(float* d_means, float* d_stddevs);
extern "C" void initRNGStates(void** d_states, int popSize, unsigned long seed);
extern "C" void freeRNGStates(void* d_states);

class GpuEvaluator {
private:
    float* d_means = nullptr;
    float* d_stddevs = nullptr;
    void* d_rngStates = nullptr; // void* bo curandState nie jest widoczne w .cpp

    int currentPopSize = 0;
    int numCities;

public:
    GpuEvaluator(const TSPProblem& problem, int maxPopSize) {
        numCities = problem.getNumCities();
        currentPopSize = maxPopSize;

        // Kopiuj dane problemu na GPU raz na początku
        allocateProblemOnGPU(problem.getMeansRaw(), problem.getStdDevsRaw(), numCities, &d_means, &d_stddevs);

        // Inicjalizuj generatory losowe na GPU
        initRNGStates(&d_rngStates, maxPopSize, 1234);
    }

    ~GpuEvaluator() {
        freeProblemOnGPU(d_means, d_stddevs);
        freeRNGStates(d_rngStates);
    }

    void evaluatePopulation(std::vector<Individual>& population, int numSimulations) {
        int popSize = population.size();

        // 1. Spłaszczanie populacji (Serialization)
        // To wykonuje się na CPU i może być kosztowne, ale konieczne
        std::vector<int> flatPopulation;
        flatPopulation.reserve(popSize * numCities);

        for (const auto& ind : population) {
            flatPopulation.insert(flatPopulation.end(), ind.path.begin(), ind.path.end());
        }

        // 2. Przygotuj miejsce na wyniki
        std::vector<float> fitnessResults(popSize);

        // 3. Uruchom Kernel
        launchEvaluation(flatPopulation.data(), d_means, d_stddevs, fitnessResults.data(), d_rngStates, numCities, popSize, numSimulations);

        // 4. Przypisz wyniki z powrotem do obiektów
        for (int i = 0; i < popSize; ++i) {
            population[i].fitness = fitnessResults[i];
        }
    }
};

#endif //USE_CUDA


#endif //STOCHASTICTSP_GPUEVALUATOR_H