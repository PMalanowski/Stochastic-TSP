#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include "TSPProblem.h"
#include "Individual.h"

// --- UPROSZCZONA KLASA GA (CZYSTA SEKWENCYJNA) ---
class SequentialGA {
private:
    const TSPProblem& problem;
    std::vector<Individual> population;
    std::vector<Individual> newPopulation;
    std::mt19937 rng;
    int popSize;
    float mutationRate = 0.02f;
    float crossoverRate = 0.9f;
    int tournamentSize = 5;

public:
    SequentialGA(const TSPProblem& p, int size, int seed) 
        : problem(p), popSize(size), rng(seed) {
        population.reserve(popSize);
        newPopulation.reserve(popSize);
        
        // Inicjalizacja
        for(int i=0; i<popSize; ++i) {
            population.emplace_back(problem.getNumCities(), rng);
            population.back().evaluateDeterministic(problem);
        }
    }

    // Prosta selekcja
    int runTournament() {
        std::uniform_int_distribution<int> dist(0, popSize - 1);
        int bestIdx = dist(rng);
        float bestFit = population[bestIdx].fitness;
        for (int i = 1; i < tournamentSize; ++i) {
            int idx = dist(rng);
            if (population[idx].fitness < bestFit) {
                bestFit = population[idx].fitness;
                bestIdx = idx;
            }
        }
        return bestIdx;
    }

    // Krzyżowanie OX
    Individual crossoverOX(const Individual& p1, const Individual& p2) {
        int N = problem.getNumCities();
        Individual child(N);
        std::uniform_int_distribution<int> dist(0, N - 1);
        int start = dist(rng);
        int end = dist(rng);
        if (start > end) std::swap(start, end);

        std::vector<bool> visited(N, false);
        for (int i = start; i <= end; ++i) {
            child.path[i] = p1.path[i];
            visited[p1.path[i]] = true;
        }
        int curr = (end + 1) % N;
        int p2Idx = (end + 1) % N;
        for (int i = 0; i < N; ++i) {
            int city = p2.path[p2Idx];
            if (!visited[city]) {
                child.path[curr] = city;
                curr = (curr + 1) % N;
            }
            p2Idx = (p2Idx + 1) % N;
        }
        return child;
    }

    // Mutacja
    void mutate(Individual& ind) {
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        if (prob(rng) < mutationRate) {
            std::uniform_int_distribution<int> dist(0, problem.getNumCities() - 1);
            int i = dist(rng);
            int j = dist(rng);
            std::swap(ind.path[i], ind.path[j]);
        }
    }

    // --- GŁÓWNA PĘTLA (BEZ OPENMP, BEZ CUDA) ---
    void runGeneration(bool useMonteCarlo, int kSamples) {
        newPopulation.clear();
        
        // Elityzm
        auto bestIt = std::min_element(population.begin(), population.end(), 
            [](const Individual& a, const Individual& b){ return a.fitness < b.fitness; });
        newPopulation.push_back(*bestIt);

        std::uniform_real_distribution<float> prob(0.0f, 1.0f);

        // Tworzenie nowej populacji
        while(newPopulation.size() < popSize) {
            int p1 = runTournament();
            int p2 = runTournament();
            Individual child(problem.getNumCities());
            
            if (prob(rng) < crossoverRate) child = crossoverOX(population[p1], population[p2]);
            else child = population[p1];

            mutate(child);

            // EWALUACJA SEKWENCYJNA
            if (useMonteCarlo) {
                // To wywołuje metodę z Individual.h (musisz ją tam mieć!)
                // Przekazujemy ten sam rng, bo jesteśmy na 1 wątku
                child.evaluateMonteCarlo(problem, kSamples, rng);
            } else {
                child.evaluateDeterministic(problem);
            }
            
            newPopulation.push_back(std::move(child));
        }
        population = newPopulation;
    }

    float getBestFitness() {
        float best = 1e9;
        for(auto& ind : population) if(ind.fitness < best) best = ind.fitness;
        return best;
    }
};

int main(int argc, char** argv) {
    // Domyślne parametry
    int N = 1000;    // Miasta
    int P = 100;    // Populacja
    int G = 100;     // Generacje
    int K = 1000;   // Symulacje MC
    // Obsługa argumentów: ./StochasticTSP_Seq N P G K
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) P = std::atoi(argv[2]);
    if (argc > 3) G = std::atoi(argv[3]);
    if (argc > 4) K = std::atoi(argv[4]);

    std::cout << "--- SEQUENTIAL BENCHMARK ---" << std::endl;
    std::cout << "Cities: " << N << ", Pop: " << P << ", Gen: " << G << ", MC Samples: " << K << std::endl;
    
    TSPProblem problem(N, 42);

    // 1. TEST DETERMINISTYCZNY
    if (K == 0)
    {
        std::cout << "\n[1] Running Deterministic (Relaxed)..." << std::endl;
        SequentialGA ga(problem, P, 1234);
        
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<G; ++i) ga.runGeneration(false, 0);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
        std::cout << "Best Fitness: " << ga.getBestFitness() << std::endl;
    } else
    {
        std::cout << "\n[2] Running Stochastic (Monte Carlo on CPU)..." << std::endl;
        SequentialGA ga(problem, P, 1234);
        
        auto start = std::chrono::high_resolution_clock::now();
        for(int i=0; i<G; ++i) ga.runGeneration(true, K);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
        std::cout << "Best Fitness: " << ga.getBestFitness() << std::endl;
    }

    return 0;
}