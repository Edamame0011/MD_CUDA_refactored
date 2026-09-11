#pragma once

#include <md/observers/Observer.hpp>
#include <memory>
#include <vector>
#include <fstream>

namespace md{    
    class Interaction;
    class State;
    class Cell;
    
    namespace observers{
        class TrajectoryExporter;
    
        class LogTrajectoryObserver : public Observer {
            public:
                LogTrajectoryObserver(float interval, int counter, bool is_unwrap, State* state, Cell& cell, Interaction* interaction, const std::string& output_path, const std::string& temp_path, const std::vector<std::string>& species_to_symbol);

                void output(State* state, SimState& simstate) override;
                void init(State* state, SimState& simstate) override;
            private:
                float log_interval_;
                int counter_;
                float checker_;
                bool is_unwarp_;
                std::unique_ptr<TrajectoryExporter> exporter_;
                std::ofstream ofs_;
        };
    }
}