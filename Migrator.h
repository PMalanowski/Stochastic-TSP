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

    // Wymiana najlepszych osobników w pierścieniu
    void exchangeBest(std::vector<Individual>& population) {
        if (size <= 1) return; // Brak migracji jeśli tylko 1 proces

        // 1. Znajdź najlepszego u siebie
        auto bestIt = std::min_element(population.begin(), population.end(),
             [](const Individual& a, const Individual& b){ return a.fitness < b.fitness; });

        // Przygotuj bufor do wysłania (sama ścieżka wystarczy, fitness przeliczy odbiorca)
        std::vector<int> sendBuffer = bestIt->path;
        std::vector<int> recvBuffer(numCities);

        // 2. MPI_Sendrecv - Wysyłamy do NEXT, odbieramy od PREV
        // To jest bezpieczne i zapobiega zakleszczeniom (deadlock) w pierścieniu
        MPI_Status status;
        MPI_Sendrecv(
            sendBuffer.data(), numCities, MPI_INT, nextRank, 0, // Send params
            recvBuffer.data(), numCities, MPI_INT, prevRank, 0, // Recv params
            MPI_COMM_WORLD, &status
        );

        // 3. Wstaw odebranego osobnika do populacji
        // Zastępujemy najgorszego osobnika, żeby nie psuć elityzmu
        auto worstIt = std::max_element(population.begin(), population.end(),
             [](const Individual& a, const Individual& b){ return a.fitness < b.fitness; });

        // Nadpisujemy ścieżkę najgorszego
        worstIt->path = recvBuffer;
        // Fitness jest nieaktualny, zostanie przeliczony w następnej generacji
        // Ustawiamy na 'nieskończoność' lub flagę, by wymusić ewaluację?
        // W naszej pętli GA ewaluacja jest na końcu, więc musimy to zrobić ręcznie teraz
        // Ale prościej: w następnej generacji zostanie wybrany do rodzicielstwa lub nie,
        // a jego fitness zostanie zaktualizowany przy następnej okazji?
        // W naszym kodzie GA 'population' jest już ocenione.
        // Więc musimy nadać mu "bezpieczny" fitness, żeby przetrwał do następnej rundy selekcji.
        worstIt->fitness = 999999.0f; // Tymczasowo słaby, ale w GA zostanie oceniony jeśli przetrwa?
        // Wróć. Najlepiej byłoby ocenić go od razu, ale nie mamy dostępu do 'problem' tutaj łatwo.
        // Zostawmy go z wysokim fitnessem, w następnym kroku GA i tak tworzymy nową populację z rodziców.
        // Jeśli chcemy, by imigrant brał udział w reprodukcji natychmiast, powinien być dobry.

        // Zróbmy tak: W main loop, po migracji, wywołamy ewaluację dla tego jednego imigranta
        // (lub dla całej populacji jeśli to szybkie).
    }
};

#endif //STOCHASTICTSP_MIGRATOR_H