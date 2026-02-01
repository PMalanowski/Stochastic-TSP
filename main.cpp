#include <iostream>
#include <mpi.h>
#include <chrono>
#include "TSPProblem.h"
#include "GeneticAlgorithm.h"
#include "Migrator.h"
#include "SimpleProfiler.h"

struct Config {
    int numCities = 100;
    int popSize = 500;
    int generations = 200;
    int mcSamples = 1000;
    int migrationInterval = 20;
    unsigned int globalSeed = 0;
};

int main(int argc, char** argv) {
    // Inicjalizacja MPI z obsługą wielowątkowości (na wszelki wypadek dla CUDA)
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    const int testMC = false; // do testowania Monte Carlo na CPU

    // --- PARAMETRY ---
    Config config;
    if (rank == 0) {
        // Opcja: ./program N P G K Mig Seed
        if (argc > 1) config.numCities = std::atoi(argv[1]);
        if (argc > 2) config.popSize = std::atoi(argv[2]);
        if (argc > 3) config.generations = std::atoi(argv[3]);
        if (argc > 4) config.mcSamples = std::atoi(argv[4]);
        if (argc > 5) config.migrationInterval = std::atoi(argv[5]);
        if (argc > 6) {
            config.globalSeed = std::atoi(argv[6]);
        }
        if (config.globalSeed == 0) {
            std::random_device rd;
            config.globalSeed = rd();
            std::cout << "[Rank 0] Generated random seed: " << config.globalSeed << std::endl;
        }
        std::cout << "--- CONFIGURATION ---" << std::endl;
        std::cout << "Cities: " << config.numCities << ", Pop: " << config.popSize
                  << ", Gen: " << config.generations << ", MC: " << config.mcSamples
                  << ", Seed: " << config.globalSeed << std::endl;
    }
    std::cout << "[Rank " << rank << "] Connected. Creating TSP instance." <<std:: endl;

    MPI_Bcast(&config, sizeof(Config), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 1. Wspólny problem (musi mieć ten sam seed!)
    TSPProblem problem(config.numCities, 42);

    // 2. Algorytm Genetyczny
    // Każdy proces ma swój inny seed losowy, żeby eksplorować inne obszary
    GeneticAlgorithm ga(problem, config.popSize, 1234 + rank);

    // 3. Konfiguracja ról
    if (rank == 0) {
#ifdef USE_CUDA
        std::cout << "[Rank 0 - GPU Node] OpenMP Threads: " << omp_get_max_threads() << std::endl;
        ga.setMode(true, config.mcSamples);
#else
        std::cout << "[Rank "<< rank <<" - CPU ONLY Node] CUDA not compiled! Running ";
        if (testMC) { // do testow zlozonosci obliczeniowej z Monte Carlo
            ga.setMode(true, config.mcSamples);
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

    Migrator migrator(config.numCities);

    SimpleProfiler profiler;

    MPI_Barrier(MPI_COMM_WORLD); // Synchronizacja startu
    auto start = std::chrono::high_resolution_clock::now();
    double globalStart = MPI_Wtime();

    // --- GŁÓWNA PĘTLA ---
    for (int i = 0; i < config.generations; ++i) {

        profiler.start("Computation");
        ga.runGeneration();
        profiler.stop("Computation");

        // Migracja
        if (i % config.migrationInterval == 0 && size > 1) {
            // Wymiana osobników
            profiler.start("Migration (Wait + Data)");
            migrator.exchangeBest(ga.getPopulation());

            // Ważne: Po wymianie musimy przeliczyć fitness imigranta
            // (bo przyszedł z wyspy o innej funkcji celu!)
            // Dla prostoty przeliczamy całą populację (na GPU to szybkie, na CPU też bo deterministyczne)
            ga.reevaluatePopulation();
            profiler.stop("Migration (Wait + Data)");

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

    profiler.saveToCSV();

    MPI_Finalize();
    return 0;
}