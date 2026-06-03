#include <md/observers/TrajectoryExporter.cuh>
#include <md/cells/CubicCell.cuh>

using namespace md::observers;

template <typename CellType>
TrajectoryExporter<CellType>::TrajectoryExporter(State& state, const std::string& output_path, const CellType& _cell) 
: cell(_cell), atom_number_map{"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"} {
    this->ofs.open(output_path);
    if (!ofs) {
        throw std::runtime_error("出力ファイルが開けませんでした。");
    }

    size_t N = state.n_atoms;
    auto view = state.get_view();

    std::vector<int> h_atomic_numbers(N);
    cudaMemcpy(h_atomic_numbers.data(), view.atomic_numbers, N * sizeof(int), cudaMemcpyDeviceToHost);

    species.resize(N);
    for (size_t i = 0; i < N; i ++) {
        species[i] = atom_number_map[h_atomic_numbers[i] - 1];
    }

    h_pos.resize(3 * N);
    h_force.resize(3 * N);
    h_box.resize(3 * N);
}

template <typename CellType>
void TrajectoryExporter<CellType>::export_trajectory(State& state) {
    auto lattice = cell.get_lattice();
    
    size_t N = state.n_atoms;
    auto view = state.get_view();

    float* h_pos_ptr = h_pos.data();
    float* h_force_ptr = h_force.data();

    // ホスト側にコピー
    cudaMemcpy(h_pos_ptr, view.pos.x, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_ptr + N, view.pos.y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_ptr + 2 * N, view.pos.z, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_force_ptr, view.force.x, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_ptr + N, view.force.y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_force_ptr + 2 * N, view.force.z, N * sizeof(float), cudaMemcpyDeviceToHost);

    // ファイルに出力
    ofs << std::setprecision(7) << std::scientific;
    ofs << N << "\n";
    ofs << "Lattice=\"" << lattice[0][0] << " 0.0 0.0 0.0 " << lattice[1][1] << " 0.0 0.0 0.0 " << lattice[2][2] << "\" " << "Properties=species:S:1:pos:R:3:forces:R:3 energy=" << state.potential_energy << " pbc=\"T T T\"" << "\n";
    for (size_t i = 0; i < N; i ++) {
        ofs << species[i] << " "
            << h_pos[i] << " " << h_pos[N + i] << " " << h_pos[2 * N + i] << " "
            << h_force[i] << " " << h_force[N + i] << " " << h_force[2 * N + i] << "\n";
    }
}

template <typename CellType>
void TrajectoryExporter<CellType>::export_trajectory_unwrap(State& state) {
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

template class TrajectoryExporter<md::cells::CubicCell>;