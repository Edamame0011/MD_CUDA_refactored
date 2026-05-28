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
    class TrajectoryExporter {
        public: 
            TrajectoryExporter(State& state, const std::string& output_path);
            void export_trajectory(State& state);

            template <typename CellType>
            void export_trajectory_unwrap(State& state, const CellType& cell) {
                auto lattice = cell.get_lattice();

                size_t N = state.n_atoms;
                auto view = state.get_view();

                float* h_pos_ptr = h_pos.data();
                float* h_force_ptr = h_force.data();
                int* h_box_ptr = h_box.data();

                // ホスト側にコピー
                cudaMemcpy(h_pos_ptr, view.pos.x, N * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_pos_ptr + N, view.pos.y, N * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_pos_ptr + 2 * N, view.pos.z, N * sizeof(float), cudaMemcpyDeviceToHost);

                cudaMemcpy(h_force_ptr, view.force.x, N * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_force_ptr + N, view.force.y, N * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_force_ptr + 2 * N, view.force.z, N * sizeof(float), cudaMemcpyDeviceToHost);

                cudaMemcpy(h_box_ptr, view.box.x, N * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_box_ptr + N, view.box.y, N * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_box_ptr + 2 * N, view.box.z, N * sizeof(int), cudaMemcpyDeviceToHost);

                // ファイルに出力
                ofs << std::setprecision(7) << std::scientific;
                ofs << N << "\n";
                ofs << "Lattice=\"" << lattice[0][0] << " 0.0 0.0 0.0 " << lattice[1][1] << " 0.0 0.0 0.0 " << lattice[2][2] << "\" " << "Properties=species:S:1:pos:R:3:forces:R:3 energy=" << state.potential_energy << " pbc=\"F F F\"" << "\n";
                for (size_t i = 0; i < N; i ++) {
                    ofs << species[i] << " "
                        << h_pos[i] + h_box[i] * lattice[0][0] << " " 
                        << h_pos[N + i] + h_box[N + i] * lattice[1][1] << " " 
                        << h_pos[2 * N + i] + h_box[2 * N + i] * lattice[2][2] << " "
                        << h_force[i] << " " << h_force[N + i] << " " << h_force[2 * N + i] << "\n";
                }
            }

        private:
            std::ofstream ofs;
            std::vector<float> h_pos;
            std::vector<float> h_force;
            std::vector<int> h_box;
            std::vector<std::string> species;
            std::vector<std::string> atom_number_map;
    };
}

#endif