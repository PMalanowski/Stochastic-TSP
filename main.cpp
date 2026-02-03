#include <iostream>
#include <mpi.h>
#include <chrono>
#include "TSPProblem.h"
#include "GeneticAlgorithm.h"
#include "Migrator.h"
#include "SimpleProfiler.h"


struct Config {
    // struktura do trzymania parametrów
    int numCities = 100;
    int popSize = 500;
    int generations = 200;
    int mcSamples = 1000;
    int migrationInterval = 20;
    int globalSeed = 0;
};

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    const int testMC = false; // do testowania Monte Carlo na CPU

    // ustawianie parametrów
    Config config;
    if (rank == 0) {
        // do usatawiania parametrow w terminalu zamiast zmieniania w kodzie i buildowania co chwile
        // w terminalu: ./StochasticTSP N P G K Mig Seed
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


    // wszystkie wezly musza rozwiazywac te sama instancje problemu
    // wiec musimy wyslac seed (+ reszte parametrow zeby nie trzeba bylo buildowac tego za kazdym razem na obu kompach
    MPI_Bcast(&config, sizeof(Config), MPI_BYTE, 0, MPI_COMM_WORLD);
    std::cout << "[Rank " << rank << "] Connected. Creating TSP instance using seed: "<< config.globalSeed <<std:: endl;

    TSPProblem problem(config.numCities, config.globalSeed);

    // genetyczny, dodajemy rank do seeda zeby kazdy wezel w rozproszonym mial inny seed
    GeneticAlgorithm ga(problem, config.popSize, config.globalSeed + rank);

    // podzial na CPU i GPU node
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
        std::cout << "[Rank " << rank << "- CPU Node] Running CPU Deterministic Mode. OpenMP Threads: " << omp_get_max_threads() << std::endl;
        ga.setMode(false, 0); // na laptopie tylko deterministyczny
    }

    Migrator migrator(config.numCities);

    // klasa specjalnie do badania wersji rozproszonej (robi csv'ki do diagramu gantta)
    //SimpleProfiler profiler;

    MPI_Barrier(MPI_COMM_WORLD); // Synchronizacja startu
    auto start = std::chrono::high_resolution_clock::now();
    //double globalStart = MPI_Wtime();

    // glowna petla
    for (int i = 0; i < config.generations; ++i) {

        // start/stop do badania rozproszonego (ile czasu liczy)
        //profiler.start("Computation");
        ga.runGeneration();
        //profiler.stop("Computation");

        // migracja (co migrationInterval generacji sie wysyla)
        if (i % config.migrationInterval == 0 && size > 1) {
            // start/stop do badania rozproszonego (ile czasu trwa migracja / ile czasu czeka na dane)
            //profiler.start("Migration (Wait + Data)");
            migrator.exchangeBest(ga.getPopulation());

            // po wymianie trzeba przeliczyc fitness bo na innym wezle jest inna funkcja celu
            ga.reevaluatePopulation();
            //profiler.stop("Migration (Wait + Data)");

            // printujemy tylko wyniki z GPU node (przy kazdej migracji albo co 50 generacji)
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
         // wynik na CPU bedzie zazwyczaj "lepszy" niz na GPU ale to dlatego, ze ma uproszczona funkcje celu
         std::cout << "--- FINAL RESULTS RANK 1 (CPU) ---" << std::endl;
         std::cout << "Best Fitness (Deterministic): " << ga.getBestFitness() << std::endl;
    }

    // do zrobienia wykresu gantta
    //profiler.saveToCSV();

    MPI_Finalize();
    return 0;
}