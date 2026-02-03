#ifndef STOCHASTICTSP_MIGRATOR_H
#define STOCHASTICTSP_MIGRATOR_H

#include <mpi.h>
#include <vector>
#include <algorithm>
#include "Individual.h"
#include "GeneticAlgorithm.h"

class Migrator {
private:
    int rank;
    int size;
    int prevRank;
    int nextRank;
    int numCities;

public:
    Migrator(int numCities) : numCities(numCities) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Ustalanie sąsiadów w pierścieniu
        nextRank = (rank + 1) % size;
        prevRank = (rank - 1 + size) % size;
    }

    // wymiana najlepszych osobników
    void exchangeBest(std::vector<Individual>& population) {
        if (size <= 1) return; // brak migracji jeśli tylko 1 proces

        // 1. Znajdź najlepszego u siebie
        auto bestIt = std::min_element(population.begin(), population.end(),
             [](const Individual& a, const Individual& b){ return a.fitness < b.fitness; });

        // wysylamy sama sciezke bo fitness i tak trzeba przeliczyc
        std::vector<int> sendBuffer = bestIt->path;
        std::vector<int> recvBuffer(numCities);

        // MPI_Sendrecv - Wysyłamy do NEXT, odbieramy od PREV
        MPI_Status status;
        MPI_Sendrecv(
            sendBuffer.data(), numCities, MPI_INT, nextRank, 0,
            recvBuffer.data(), numCities, MPI_INT, prevRank, 0,
            MPI_COMM_WORLD, &status
        );

        // zastępujemy najgorszego osobnika tym otrzymanym
        auto worstIt = std::max_element(population.begin(), population.end(),
             [](const Individual& a, const Individual& b){ return a.fitness < b.fitness; });

        worstIt->path = recvBuffer;
        worstIt->fitness = 999999.0f;
    }
};

#endif //STOCHASTICTSP_MIGRATOR_H