#include <iostream>
#include <mpi.h>
#include <chrono>
#include "TSPProblem.h"
#include "GeneticAlgorithm.h"
#include "Migrator.h"

int main(int argc, char** argv) {
    // Inicjalizacja MPI z obsługą wielowątkowości (na wszelki wypadek dla CUDA)
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- PARAMETRY ---
    const int NUM_CITIES = 10000;
    const int POP_SIZE = 20;       // Nieco mniejsza populacja bo mamy wiele wysp
    const int GENERATIONS = 20;
    const int MC_SAMPLES = 100;
    const int MIGRATION_INTERVAL = 20; // Co ile generacji wymiana
    const int testMC = true; // do testowania Monte Carlo na CPU

    // 1. Wspólny problem (musi mieć ten sam seed!)
    TSPProblem problem(NUM_CITIES, 42);

    // 2. Algorytm Genetyczny
    // Każdy proces ma swój inny seed losowy, żeby eksplorować inne obszary
    GeneticAlgorithm ga(problem, POP_SIZE, 1234 + rank);

    // 3. Konfiguracja ról
    if (rank == 0) {
#ifdef USE_CUDA
        std::cout << "[Rank 0 - GPU Node] OpenMP Threads: " << omp_get_max_threads() << std::endl;
        ga.setMode(true, MC_SAMPLES);
#else
        std::cout << "[Rank "<< rank <<" - CPU ONLY Node] CUDA not compiled! Running ";
        if (testMC) { // do testow zlozonosci obliczeniowej z Monte Carlo
            ga.setMode(true, MC_SAMPLES);
            std::cout << "Stochastic. ";
        } else {
            ga.setMode(false, 0);
            std::cout << "Deterministic. ";
        }
        std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
#endif
    } else {
        // SLAVE / CPU NODES
        std::cout << "[Rank " << rank << "- CPU Node] Running CPU Deterministic Mode. OpenMP Threads: " << omp_get_max_threads() << std::endl;
        ga.setMode(false, 0); // na laptopie tylko deterministyczny
    }

    Migrator migrator(NUM_CITIES);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronizacja startu
    auto start = std::chrono::high_resolution_clock::now();

    // --- GŁÓWNA PĘTLA ---
    for (int i = 0; i < GENERATIONS; ++i) {

        ga.runGeneration();

        // Migracja
        if (i % MIGRATION_INTERVAL == 0 && size > 1) {
            // Wymiana osobników
            migrator.exchangeBest(ga.getPopulation());

            // Ważne: Po wymianie musimy przeliczyć fitness imigranta
            // (bo przyszedł z wyspy o innej funkcji celu!)
            // Dla prostoty przeliczamy całą populację (na GPU to szybkie, na CPU też bo deterministyczne)
            ga.reevaluatePopulation();

            if (rank == 0) {
                std::cout << "[Rank 0] Generation " << i << " Best (MC): " << ga.getBestFitness() << " (Migration done)" << std::endl;
            }
        } else if (i % 50 == 0 && rank == 0) {
            std::cout << "[Rank 0] Generation " << i << " Best (MC): " << ga.getBestFitness() << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        std::cout << "--- FINAL RESULTS RANK 0 (GPU) ---" << std::endl;
        std::cout << "Best Fitness (95% VaR): " << ga.getBestFitness() << std::endl;
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
    }
    else if (rank == 1) {
         // Uwaga: Fitness na CPU jest liczony jako suma średnich, więc będzie niższy niż MC
         std::cout << "--- FINAL RESULTS RANK 1 (CPU) ---" << std::endl;
         std::cout << "Best Fitness (Deterministic): " << ga.getBestFitness() << std::endl;
    }

    MPI_Finalize();
    return 0;
}