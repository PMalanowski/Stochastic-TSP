#ifndef STOCHASTICTSP_GPUEVALUATOR_H
#define STOCHASTICTSP_GPUEVALUATOR_H

#ifdef USE_CUDA

#include <vector>
#include "TSPProblem.h"
#include "Individual.h"

extern "C" void launchEvaluation(const int* h_population, const float* d_means, const float* d_stddevs, float* h_fitness, void* d_states, int numCities, int popSize, int numSimulations);
extern "C" void allocateProblemOnGPU(const float* h_means, const float* h_stddevs, int numCities, float** d_means, float** d_stddevs);
extern "C" void freeProblemOnGPU(float* d_means, float* d_stddevs);
extern "C" void initRNGStates(void** d_states, int popSize, unsigned long seed);
extern "C" void freeRNGStates(void* d_states);

class GpuEvaluator {
private:
    float* d_means = nullptr;
    float* d_stddevs = nullptr;
    void* d_rngStates = nullptr;

    int currentPopSize = 0;
    int numCities;

public:
    GpuEvaluator(const TSPProblem& problem, int maxPopSize) {
        numCities = problem.getNumCities();
        currentPopSize = maxPopSize;

        // kopia danych problemu na GPU raz na początku
        allocateProblemOnGPU(problem.getMeansRaw(), problem.getStdDevsRaw(), numCities, &d_means, &d_stddevs);

        // generatory losowe na GPU
        initRNGStates(&d_rngStates, maxPopSize, 1234);
    }

    ~GpuEvaluator() {
        freeProblemOnGPU(d_means, d_stddevs);
        freeRNGStates(d_rngStates);
    }

    void evaluatePopulation(std::vector<Individual>& population, int numSimulations) {
        int popSize = population.size();

        std::vector<int> flatPopulation;
        flatPopulation.reserve(popSize * numCities);

        for (const auto& ind : population) {
            flatPopulation.insert(flatPopulation.end(), ind.path.begin(), ind.path.end());
        }

        std::vector<float> fitnessResults(popSize);

        launchEvaluation(flatPopulation.data(), d_means, d_stddevs, fitnessResults.data(), d_rngStates, numCities, popSize, numSimulations);

        for (int i = 0; i < popSize; ++i) {
            population[i].fitness = fitnessResults[i];
        }
    }
};

#endif //USE_CUDA


#endif //STOCHASTICTSP_GPUEVALUATOR_H