#ifndef STOCHASTICTSP_INDIVIDUAL_H
#define STOCHASTICTSP_INDIVIDUAL_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include "TSPProblem.h"

class Individual {
public:
    std::vector<int> path; // sciezka (permutacja miast)
    float fitness;

    // Konstruktor tworzący losowego osobnika
    Individual(int numCities, std::mt19937& gen) {
        path.resize(numCities);
        std::iota(path.begin(), path.end(), 0); // Wypełnij 0, 1, 2...
        std::shuffle(path.begin(), path.end(), gen); // Losowa permutacja
        fitness = 0.0f;
    }

    // Konstruktor pusty (dla dzieci z krzyżowania)
    Individual(int numCities) {
        path.resize(numCities);
        fitness = 0.0f;
    }

    // ocena dla zrelaksowanej funkcji
    // vvvv odkomentowac zeby sie wyswietlalo w profilerze vvvv
    //__attribute__((noinline))
    void evaluateDeterministic(const TSPProblem& problem) {
        float sum = 0.0f;
        int numCities = problem.getNumCities();
        for (int i = 0; i < numCities - 1; ++i) {
            sum += problem.getMean(path[i], path[i+1]);
        }
        sum += problem.getMean(path[numCities-1], path[0]);
        fitness = sum;
    }

    // ocena dla rozszerzonej funkcji celu
    void evaluateMonteCarlo(const TSPProblem& problem, int k, std::mt19937& globalGen) {
        std::vector<float> simulationResults;
        simulationResults.reserve(k);
        int numCities = problem.getNumCities();

        // generujemy standardowy rozkład normalny N(0,1) i skalujemy: mean + std * rand
        std::normal_distribution<float> standardNormal(0.0f, 1.0f);

        for (int sim = 0; sim < k; ++sim) {
            float routeTime = 0.0f;

            // przejście przez całą trasę
            for (int i = 0; i < numCities; ++i) {
                int u = path[i];
                int v = path[(i + 1) % numCities];

                float mu = problem.getMean(u, v);
                float sigma = problem.getStdDev(u, v);

                // losowanie czasu dla tego odcinka: N(mu, sigma)
                float edgeTime = mu + sigma * standardNormal(globalGen);

                if (edgeTime < 0) edgeTime = 0.001f;

                routeTime += edgeTime;
            }
            simulationResults.push_back(routeTime);
        }

        // obliczamy fitness jako 95-ty percentyl
        std::sort(simulationResults.begin(), simulationResults.end());

        int idx = (int)(0.95 * k);
        if (idx >= k) idx = k - 1;

        fitness = simulationResults[idx];
    }
};

#endif //STOCHASTICTSP_INDIVIDUAL_H