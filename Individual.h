#ifndef STOCHASTICTSP_INDIVIDUAL_H
#define STOCHASTICTSP_INDIVIDUAL_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include "TSPProblem.h"

class Individual {
public:
    std::vector<int> path; // Permutacja indeksów miast
    float fitness;         // Wartość funkcji celu (im mniej tym lepiej)

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

    // Prosta ocena deterministyczna (dla CPU / Weak Node)
    // Suma średnich czasów
    //__attribute__((noinline))
    void evaluateDeterministic(const TSPProblem& problem) {
        float sum = 0.0f;
        int numCities = problem.getNumCities();
        for (int i = 0; i < numCities - 1; ++i) {
            sum += problem.getMean(path[i], path[i+1]);
        }
        // Powrót do miasta startowego
        sum += problem.getMean(path[numCities-1], path[0]);
        fitness = sum;
    }
    // Nowa metoda: Symulacja Monte Carlo
    // k = liczba symulacji dla jednej trasy (np. 1000)
    void evaluateMonteCarlo(const TSPProblem& problem, int k, std::mt19937& globalGen) {
        std::vector<float> simulationResults;
        simulationResults.reserve(k);
        int numCities = problem.getNumCities();

        // Obiekty rozkładów tworzymy raz, ale parametry zmieniamy w pętli?
        // Szybciej: generujemy standardowy rozkład normalny N(0,1) i skalujemy: mean + std * rand
        std::normal_distribution<float> standardNormal(0.0f, 1.0f);

        for (int sim = 0; sim < k; ++sim) {
            float routeTime = 0.0f;

            // Przejście przez całą trasę
            for (int i = 0; i < numCities; ++i) {
                int u = path[i];
                int v = path[(i + 1) % numCities]; // Powrót do startu na końcu

                float mu = problem.getMean(u, v);
                float sigma = problem.getStdDev(u, v);

                // Losowanie czasu dla tego odcinka: N(mu, sigma)
                float edgeTime = mu + sigma * standardNormal(globalGen);

                // Czas nie może być ujemny (zabezpieczenie matematyczne)
                if (edgeTime < 0) edgeTime = 0.001f;

                routeTime += edgeTime;
            }
            simulationResults.push_back(routeTime);
        }

        // Obliczamy fitness jako 95-ty percentyl (pesymistyczny wariant)
        // Sortujemy wyniki symulacji
        std::sort(simulationResults.begin(), simulationResults.end());

        // Indeks dla 95%
        int idx = (int)(0.95 * k);
        if (idx >= k) idx = k - 1;

        fitness = simulationResults[idx];
    }
};

#endif //STOCHASTICTSP_INDIVIDUAL_H