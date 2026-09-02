#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace md {
    class Cell;
    struct State;
}

namespace md::observers {
    // Extended XYZ writer.  Frames are always restored to the original
    // particle order using State::particle_id.
    class TrajectoryExporter {
        public:
            // particle_id and image must have been initialized by the caller.
            TrajectoryExporter(const State& state, const std::string& output_path, Cell* cell);

            void export_trajectory(const State& state);
            void export_trajectory_unwrap(const State& state);

        private:
            void export_frame(const State& state, bool unwrap);

            std::ofstream output_;
            Cell* cell_;
            std::vector<float> positions_;
            std::vector<float> forces_;
            std::vector<int> images_;
            std::vector<int> species_;
            std::vector<int> particle_ids_;
    };
}
