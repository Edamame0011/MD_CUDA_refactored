#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace md {
    class Cell;
    struct State;
    struct SimState;
    class Interaction;
}

namespace md::observers {
    // Extended XYZ writer.  Frames are always restored to the original
    // particle order using State::particle_id.
    class TrajectoryExporter {
        public:
            // particle_id and image must have been initialized by the caller.
            TrajectoryExporter(State* state, Cell& cell, Interaction* interaction, const std::string& output_path, const std::vector<std::string>& species_to_symbol);
            
            void export_frame(State* state, SimState& simstate, bool unwrap);

        private:
            std::ofstream output_;
            Cell& cell_;
            std::vector<std::string> species_to_symbol_;
            std::vector<float> positions_;
            std::vector<float> forces_;
            std::vector<int> images_;
            std::vector<int> species_;
            std::vector<int> particle_ids_;
            Interaction* interaction_;
    };
}
