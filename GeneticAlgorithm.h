#ifndef STOCHASTICTSP_GENETICALGORITHM_H
#define STOCHASTICTSP_GENETICALGORITHM_H

#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <omp.h> // Biblioteka OpenMP
#include "TSPProblem.h"
#include "Individual.h"
#include "GpuEvaluator.h"

class GeneticAlgorithm {
private:
    const TSPProblem& problem;
    std::vector<Individual> population;
    std::vector<Individual> newPopulation;

#ifdef USE_CUDA
    GpuEvaluator* gpuEvaluator = nullptr;
#endif

    int popSize;
    float mutationRate;
    float crossoverRate;
    int tournamentSize;

    bool useMonteCarlo = false;
    int mcSimulations = 100;

    // wektor generatorow bo kazdy watek dostanie swoj
    std::vector<std::mt19937> threadRNGs;

public:
    GeneticAlgorithm(const TSPProblem& p, int populationSize, int seed = 42)
        : problem(p), popSize(populationSize)
    {
        mutationRate = 0.02f;
        crossoverRate = 0.9f;
        tournamentSize = 5;

#ifdef USE_CUDA
        gpuEvaluator = new GpuEvaluator(problem, popSize);
#endif

        //  ustawienie generatorów losowych (jeden na wątek)
        int maxThreads = omp_get_max_threads();
        if (maxThreads < 1) maxThreads = 1;
        threadRNGs.resize(maxThreads);

        for(int i=0; i<maxThreads; ++i) {
            threadRNGs[i].seed(seed + i * 9999);
        }

        // rezerwacja pamieci na populacje od razu
        population.resize(popSize, Individual(problem.getNumCities()));
        newPopulation.resize(popSize, Individual(problem.getNumCities()));

        // rownolegle wypelnianie populacji
#pragma omp parallel for
        for (int i = 0; i < popSize; ++i) {
            int tid = omp_get_thread_num();
            if (tid >= maxThreads) tid = 0;

            population[i] = Individual(problem.getNumCities(), threadRNGs[tid]);

            // startowa populacje od razu oceniamy (deterministyczna funkcja celu dla uproszczenia)
            population[i].evaluateDeterministic(problem);
        }
    }

#ifdef USE_CUDA
    ~GeneticAlgorithm() {
        if (gpuEvaluator) {
            delete gpuEvaluator;
            gpuEvaluator = nullptr;
        }
    }
#endif
    // Zmodyfikowana metoda turnieju, przyjmuje konkretny RNG
    int runTournament(std::mt19937& localRng) {
        std::uniform_int_distribution<int> dist(0, popSize - 1);
        int bestIdx = dist(localRng);
        float bestFit = population[bestIdx].fitness;

        for (int i = 1; i < tournamentSize; ++i) {
            int contenderIdx = dist(localRng);
            float contenderFit = population[contenderIdx].fitness;
            if (contenderFit < bestFit) {
                bestFit = contenderFit;
                bestIdx = contenderIdx;
            }
        }
        return bestIdx;
    }

    // Metoda krzyżowania i mutacji też muszą przyjmować RNG
    Individual crossoverOX(const Individual& p1, const Individual& p2, std::mt19937& localRng) {
        int N = problem.getNumCities();
        Individual child(N);

        // losowanie zakresu do odziedziczenia z pierwszego rodzica
        std::uniform_int_distribution<int> dist(0, N - 1);
        int start = dist(localRng);
        int end = dist(localRng);
        if (start > end) std::swap(start, end);

        std::vector<bool> visited(N, false); // wektor do przechowywania informacji czy dane miasto już jest w potomku
        for (int i = start; i <= end; ++i) {
            int city = p1.path[i];
            child.path[i] = city;
            visited[city] = true;
        }

        int currentIdx = (end + 1) % N;
        int p2Idx = (end + 1) % N;

        for (int i = 0; i < N; ++i) {
            int city = p2.path[p2Idx];
            if (!visited[city]) {
                child.path[currentIdx] = city;
                currentIdx = (currentIdx + 1) % N;
            }
            p2Idx = (p2Idx + 1) % N;
        }
        return child;
    }

    void mutate(Individual& ind, std::mt19937& localRng) {
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        if (prob(localRng) < mutationRate) {
            std::uniform_int_distribution<int> dist(0, problem.getNumCities() - 1);
            int i = dist(localRng);
            int j = dist(localRng);
            std::swap(ind.path[i], ind.path[j]);
        }
    }

    void runGeneration() {
        auto bestIt = std::min_element(population.begin(), population.end(),
            [](const Individual& a, const Individual& b){ return a.fitness < b.fitness; });

        Individual elite = *bestIt;

        // tworzenie nowej populacji (rownolegle)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < popSize; ++i) {
            // Elityzm: Pierwszy osobnik to kopia najlepszego
            if (i == 0) {
                newPopulation[i] = elite;
                continue;
            }

            int tid = omp_get_thread_num(); // ID wątku
            std::mt19937& localRng = threadRNGs[tid]; // RNG przypisany do wątku

            std::uniform_real_distribution<float> prob(0.0f, 1.0f);

            // Selekcja
            int p1Idx = runTournament(localRng);
            int p2Idx = runTournament(localRng);

            Individual child(problem.getNumCities());

            // Krzyzowanie
            if (prob(localRng) < crossoverRate) {
                child = crossoverOX(population[p1Idx], population[p2Idx], localRng);
            } else {
                child = population[p1Idx];
            }

            mutate(child, localRng);

            newPopulation[i] = std::move(child);
        }

#ifdef USE_CUDA
        if (useMonteCarlo && gpuEvaluator) {
            gpuEvaluator->evaluatePopulation(newPopulation, mcSimulations);
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < popSize; ++i) {
                newPopulation[i].evaluateDeterministic(problem);
            }
        }
#else
        // wersja dla lapka (zawsze CPU)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < popSize; ++i) {
            if (useMonteCarlo) {
                // Wersja Stochastyczna na CPU - do testów
                int tid = omp_get_thread_num();
                newPopulation[i].evaluateMonteCarlo(problem, mcSimulations, threadRNGs[tid]);
            } else {
                // Wersja Deterministyczna
                newPopulation[i].evaluateDeterministic(problem);
            }
        }
#endif

        population = newPopulation;
    }

    void setMode(bool mc, int k) {
        useMonteCarlo = mc;
        mcSimulations = k;
    }

    std::vector<Individual>& getPopulation() { return population; }

    float getBestFitness() const {
        float best = 9999999.0f;
        for(const auto& ind : population) {
            if(ind.fitness < best) best = ind.fitness;
        }
        return best;
    }

    void reevaluatePopulation() {
#ifdef USE_CUDA
        if (useMonteCarlo && gpuEvaluator) {
            gpuEvaluator->evaluatePopulation(population, mcSimulations);
        } else {
            #pragma omp parallel for schedule(static)
            for(int i=0; i<popSize; ++i) {
                population[i].evaluateDeterministic(problem);
            }
        }
#else
        #pragma omp parallel for schedule(static)
        for(int i=0; i<popSize; ++i) {
            population[i].evaluateDeterministic(problem);
        }
#endif
    }
};

#endif //STOCHASTICTSP_GENETICALGORITHM_H