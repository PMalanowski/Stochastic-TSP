#ifndef STOCHASTICTSP_SIMPLEPROFILER_H
#define STOCHASTICTSP_SIMPLEPROFILER_H

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <mpi.h>

struct ProfileEvent {
    std::string name;
    double start;
    double end;
    int rank;
};

class SimpleProfiler {
private:
    std::vector<ProfileEvent> events;
    std::map<std::string, double> activeTimers;
    int rank;

public:
    SimpleProfiler() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    void start(const std::string& name) {
        activeTimers[name] = MPI_Wtime();
    }

    void stop(const std::string& name) {
        double endTime = MPI_Wtime();
        if (activeTimers.find(name) != activeTimers.end()) {
            events.push_back({name, activeTimers[name], endTime, rank});
            activeTimers.erase(name);
        }
    }

    void saveToCSV() {
        std::string filename = "profile_rank_" + std::to_string(rank) + ".csv";
        std::ofstream file(filename);

        // Nagłówek: Rank, EventName, StartTime, EndTime, Duration
        file << "Rank,Event,Start,End,Duration\n";

        for (const auto& e : events) {
            file << e.rank << ","
                 << e.name << ","
                 << e.start << ","
                 << e.end << ","
                 << (e.end - e.start) << "\n";
        }
        file.close();
    }
};

#endif //STOCHASTICTSP_SIMPLEPROFILER_H