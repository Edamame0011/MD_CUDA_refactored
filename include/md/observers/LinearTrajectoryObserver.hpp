#pragma once

#include <md/observers/Observer.hpp>
#include <memory>
#include <vector>

namespace md{    
    class Interaction;
    class State;
    class Cell;
    
    namespace observers{
        class TrajectoryExporter;
    
        class LinearTrajectoryObserver : public Observer {
            public:
                LinearTrajectoryObserver(int interval, bool is_unwrap, State* state, Cell& cell, Interaction* interaction, const std::string& output_path, const std::vector<std::string>& species_to_symbol);

                void output(State* state, SimState& simstate) override;
                void init(State* state, SimState& simstate) override;
            private:
                int output_interval_;
                bool is_unwarp_;
                std::unique_ptr<TrajectoryExporter> exporter_;
        };
    }
}