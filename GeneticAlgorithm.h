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

    // Zamiast jednego RNG, mamy wektor RNG (jeden dla każdego wątku OpenMP)
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

        // --- INICJALIZACJA OPENMP I RNG ---
        int maxThreads = omp_get_max_threads();
        threadRNGs.resize(maxThreads);

        // Każdy wątek dostaje inny seed bazując na głównym seedzie + id wątku
        for(int i=0; i<maxThreads; ++i) {
            threadRNGs[i].seed(seed + i * 9999);
        }

        population.reserve(popSize);
        newPopulation.resize(popSize, Individual(problem.getNumCities())); // Resize zamiast reserve dla OpenMP!

        // Inicjalizacja populacji (można zrównoleglić)
        #pragma omp parallel for
        for (int i = 0; i < popSize; ++i) {
            int tid = omp_get_thread_num(); // Pobierz ID wątku
            population.emplace_back(problem.getNumCities(), threadRNGs[tid]); // Użyj RNG tego wątku (nie działa z emplace_back dynamicznie dobrze w parallel for bez pre-alokacji)
        }

        // Poprawka inicjalizacji:
        population.clear();
        population.resize(popSize, Individual(problem.getNumCities()));

        #pragma omp parallel for
        for(int i=0; i<popSize; ++i) {
             int tid = omp_get_thread_num();
             // Nadpisz pustego osobnika nowym losowym
             population[i] = Individual(problem.getNumCities(), threadRNGs[tid]);
             population[i].evaluateDeterministic(problem);
        }
    }

    ~GeneticAlgorithm() {
        if (gpuEvaluator) {
            delete gpuEvaluator;
            gpuEvaluator = nullptr;
        }
    }

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

        std::uniform_int_distribution<int> dist(0, N - 1);
        int start = dist(localRng);
        int end = dist(localRng);
        if (start > end) std::swap(start, end);

        std::vector<bool> visited(N, false);
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

    // --- RÓWNOLEGŁA PĘTLA GENERACJI ---
    void runGeneration() {
        // 1. Elityzm (sekwencyjnie, to jest szybkie)
        auto bestIt = std::min_element(population.begin(), population.end(),
            [](const Individual& a, const Individual& b){ return a.fitness < b.fitness; });

        Individual elite = *bestIt; // Kopia elity

        // 2. Równoległe tworzenie nowej populacji (Breeding Phase)
        // Używamy "static schedule" dla przewidywalności lub "dynamic" jeśli czasy operacji się różnią
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < popSize; ++i) {
            // Elityzm: Pierwszy osobnik to kopia najlepszego
            if (i == 0) {
                newPopulation[i] = elite;
                continue; // Przejdź do następnej iteracji pętli
            }

            int tid = omp_get_thread_num(); // ID wątku (0...N-1)
            std::mt19937& localRng = threadRNGs[tid]; // RNG przypisany do wątku

            std::uniform_real_distribution<float> prob(0.0f, 1.0f);

            // Selekcja
            int p1Idx = runTournament(localRng);
            int p2Idx = runTournament(localRng);

            Individual child(problem.getNumCities());

            // Krzyżowanie
            if (prob(localRng) < crossoverRate) {
                child = crossoverOX(population[p1Idx], population[p2Idx], localRng);
            } else {
                child = population[p1Idx];
            }

            // Mutacja
            mutate(child, localRng);

            // Zapisz dziecko w pre-alokowanej tablicy (bezpieczne wątkowo dzięki indeksowi 'i')
            newPopulation[i] = std::move(child);
        }

#ifdef USE_CUDA
        if (useMonteCarlo && gpuEvaluator) {
            gpuEvaluator->evaluatePopulation(newPopulation, mcSimulations);
        } else {
            // Fallback do CPU jeśli włączono MC ale coś poszło nie tak, lub standardowa ścieżka CPU
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < popSize; ++i) {
                newPopulation[i].evaluateDeterministic(problem);
            }
        }
#else
        // Wersja dla Laptopa (zawsze CPU)
#pragma omp parallel for schedule(static)
        for (int i = 0; i < popSize; ++i) {
            newPopulation[i].evaluateDeterministic(problem);
        }
#endif

        // 4. Podmiana populacji
        population = newPopulation;
    }

    // Gettery i settery bez zmian...
    void setMode(bool mc, int k) {
        useMonteCarlo = mc;
        mcSimulations = k;
    }

    std::vector<Individual>& getPopulation() { return population; }

    float getBestFitness() const {
        float best = 9999999.0f;
        // Redukcja min w OpenMP jest możliwa, ale dla bezpieczeństwa i prostoty zróbmy liniowo
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
#pragma omp parallel for
            for(int i=0; i<popSize; ++i) ind.evaluateDeterministic(problem);
        }
#else
        #pragma omp parallel for
        for(int i=0; i<popSize; ++i) population[i].evaluateDeterministic(problem);
#endif
    }
};

#endif //STOCHASTICTSP_GENETICALGORITHM_H