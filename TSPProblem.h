#ifndef STOCHASTICTSP_TSPPROBLEM_H
#define STOCHASTICTSP_TSPPROBLEM_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

struct City {
    int id;
    float x, y;
};

class TSPProblem {
private:
    std::vector<City> cities;
    // Macierze są spłaszczone do 1D dla wydajności (row-major order)
    // Dostęp: matrix[i * numCities + j]
    std::vector<float> means;   // Średnie czasy przejazdu
    std::vector<float> stddevs; // Odchylenia standardowe

public:
    TSPProblem(int numCities, unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> coordDist(0.0f, 100.0f);
        // Większe odchylenie = trudniejszy problem stochastyczny
        std::uniform_real_distribution<float> stdDevFactor(0.05f, 0.30f);

        cities.resize(numCities);
        means.resize(numCities * numCities);
        stddevs.resize(numCities * numCities);

        // 1. Generuj miasta
        for (int i = 0; i < numCities; ++i) {
            cities[i] = {i, coordDist(gen), coordDist(gen)};
        }

        // 2. Pre-kalkuluj macierz sąsiedztwa (średnie i odchylenia)
        for (int i = 0; i < numCities; ++i) {
            for (int j = 0; j < numCities; ++j) {
                if (i == j) {
                    means[i * numCities + j] = 0.0f;
                    stddevs[i * numCities + j] = 0.0f;
                } else {
                    float dist = std::sqrt(std::pow(cities[i].x - cities[j].x, 2) +
                                           std::pow(cities[i].y - cities[j].y, 2));
                    means[i * numCities + j] = dist;
                    // Odchylenie zależy od odległości (dłuższa trasa = większa niepewność)
                    stddevs[i * numCities + j] = dist * stdDevFactor(gen);
                }
            }
        }
    }

    // Gettery (inline dla wydajności)
    int getNumCities() const { return cities.size(); }

    // Zwraca wskaźniki do surowych danych (potrzebne później dla CUDA/MPI)
    const float* getMeansRaw() const { return means.data(); }
    const float* getStdDevsRaw() const { return stddevs.data(); }

    float getMean(int i, int j) const {
        return means[i * cities.size() + j];
    }

    float getStdDev(int i, int j) const {
        return stddevs[i * cities.size() + j];
    }
};

#endif //STOCHASTICTSP_TSPPROBLEM_H