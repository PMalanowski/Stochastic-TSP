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
    std::vector<float> means;   // srednie czasy przejazdu
    std::vector<float> stddevs; // odchylenia standardowe

public:
    TSPProblem(int numCities, unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> coordDist(0.0f, 100.0f);
        std::uniform_real_distribution<float> stdDevFactor(0.05f, 0.30f);

        cities.resize(numCities);
        means.resize(numCities * numCities);
        stddevs.resize(numCities * numCities);

        // generowanie miast (x,y) w kwadracie 100x100
        for (int i = 0; i < numCities; ++i) {
            cities[i] = {i, coordDist(gen), coordDist(gen)};
        }

        // macierz sąsiedztwa (średnie i odchylenia)
        for (int i = 0; i < numCities; ++i) {
            for (int j = 0; j < numCities; ++j) {
                if (i == j) {
                    // czas przejazdu / dystans z miasta do tego samego miasta to 0
                    means[i * numCities + j] = 0.0f;
                    stddevs[i * numCities + j] = 0.0f;
                } else {
                    float dist = std::sqrt(std::pow(cities[i].x - cities[j].x, 2) +
                                           std::pow(cities[i].y - cities[j].y, 2));
                    means[i * numCities + j] = dist;
                    // losowanie odchylenia standardowego dla drogi miedzy i-j
                    stddevs[i * numCities + j] = dist * stdDevFactor(gen);
                }
            }
        }
    }

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