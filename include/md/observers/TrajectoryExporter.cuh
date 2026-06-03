#ifndef __TRAJECTORY_EXPORTER_CUH__
#define __TRAJECTORY_EXPORTER_CUH__

#include <md/core/State.cuh>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <iomanip>

namespace md::observers {
    template <typename CellType>
    class TrajectoryExporter {
        public: 
            TrajectoryExporter(State& state, const std::string& output_path, const CellType& cell);
            void export_trajectory(State& state);

            void export_trajectory_unwrap(State& state);

        private:
            std::ofstream ofs;
            std::vector<float> h_pos;
            std::vector<float> h_force;
            std::vector<int> h_box;
            std::vector<std::string> species;
            std::vector<std::string> atom_number_map;
            CellType cell;
    };
}

#endif