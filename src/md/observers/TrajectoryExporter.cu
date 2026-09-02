#include <md/observers/TrajectoryExporter.hpp>

#include <md/core/Cell.cuh>
#include <md/core/State.hpp>

#include <cuda_runtime.h>

#include <iomanip>
#include <stdexcept>

namespace {
    std::string species_label(int species) {
        // The current State stores species indices, not atomic numbers.
        if (species < 0) {
            throw std::runtime_error("TrajectoryExporter: negative species ID " + std::to_string(species));
        }
        return "X" + std::to_string(species);
    }

    void check_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("TrajectoryExporter: ") + operation + ": " + cudaGetErrorString(status));
        }
    }
}

namespace md::observers {
    TrajectoryExporter::TrajectoryExporter(
        const State& state,
        const std::string& output_path,
        Cell* cell
    ) : output_(output_path), cell_(cell) {
        if (cell_ == nullptr) {
            throw std::invalid_argument("TrajectoryExporter: cell must not be null");
        }
        if (!output_) {
            throw std::runtime_error("TrajectoryExporter: could not open output file: " + output_path);
        }

        const auto n = static_cast<std::size_t>(state.n_atoms);
        positions_.resize(3 * n);
        forces_.resize(3 * n);
        images_.resize(3 * n);
        species_.resize(n);
        particle_ids_.resize(n);
    }

    void TrajectoryExporter::export_trajectory(const State& state) {
        export_frame(state, false);
    }

    void TrajectoryExporter::export_trajectory_unwrap(const State& state) {
        export_frame(state, true);
    }

    void TrajectoryExporter::export_frame(const State& state, bool unwrap) {
        const auto n = static_cast<std::size_t>(state.n_atoms);
        if (positions_.size() != 3 * n) {
            throw std::invalid_argument("TrajectoryExporter: atom count changed after construction");
        }

        check_cuda(cudaMemcpy(positions_.data(), state.pos.x, n * sizeof(float), cudaMemcpyDeviceToHost), "copy x positions");
        check_cuda(cudaMemcpy(positions_.data() + n, state.pos.y, n * sizeof(float), cudaMemcpyDeviceToHost), "copy y positions");
        check_cuda(cudaMemcpy(positions_.data() + 2 * n, state.pos.z, n * sizeof(float), cudaMemcpyDeviceToHost), "copy z positions");
        check_cuda(cudaMemcpy(forces_.data(), state.force.x, n * sizeof(float), cudaMemcpyDeviceToHost), "copy x forces");
        check_cuda(cudaMemcpy(forces_.data() + n, state.force.y, n * sizeof(float), cudaMemcpyDeviceToHost), "copy y forces");
        check_cuda(cudaMemcpy(forces_.data() + 2 * n, state.force.z, n * sizeof(float), cudaMemcpyDeviceToHost), "copy z forces");
        check_cuda(cudaMemcpy(species_.data(), state.species, n * sizeof(int), cudaMemcpyDeviceToHost), "copy species");
        check_cuda(cudaMemcpy(particle_ids_.data(), state.particle_id, n * sizeof(int), cudaMemcpyDeviceToHost), "copy particle IDs");
        if (unwrap) {
            check_cuda(cudaMemcpy(images_.data(), state.image.x, n * sizeof(int), cudaMemcpyDeviceToHost), "copy x images");
            check_cuda(cudaMemcpy(images_.data() + n, state.image.y, n * sizeof(int), cudaMemcpyDeviceToHost), "copy y images");
            check_cuda(cudaMemcpy(images_.data() + 2 * n, state.image.z, n * sizeof(int), cudaMemcpyDeviceToHost), "copy z images");
        }

        std::vector<int> current_index_by_id(n, -1);
        for (std::size_t current = 0; current < n; ++current) {
            const int id = particle_ids_[current];
            if (id < 0 || id >= static_cast<int>(n) || current_index_by_id[id] != -1) {
                throw std::runtime_error("TrajectoryExporter: particle_id is not a permutation of [0, n_atoms)");
            }
            current_index_by_id[id] = static_cast<int>(current);
        }

        const auto lattice = cell_->get_lattice();
        output_ << std::scientific << std::setprecision(7);
        output_ << n << '\n';
        output_ << "Lattice=\"" << lattice[0] << " 0.0 0.0 0.0 "
                << lattice[1] << " 0.0 0.0 0.0 " << lattice[2]
                << "\" Properties=species:S:1:pos:R:3:forces:R:3 pbc=\""
                << (unwrap ? "F F F" : "T T T") << "\"\n";

        for (std::size_t id = 0; id < n; ++id) {
            const auto current = static_cast<std::size_t>(current_index_by_id[id]);
            const float x = positions_[current] + (unwrap ? images_[current] * lattice[0] : 0.0f);
            const float y = positions_[n + current] + (unwrap ? images_[n + current] * lattice[1] : 0.0f);
            const float z = positions_[2 * n + current] + (unwrap ? images_[2 * n + current] * lattice[2] : 0.0f);
            output_ << species_label(species_[current]) << ' '
                    << x << ' ' << y << ' ' << z << ' '
                    << forces_[current] << ' ' << forces_[n + current] << ' ' << forces_[2 * n + current] << '\n';
        }
        output_.flush();
        if (!output_) {
            throw std::runtime_error("TrajectoryExporter: failed while writing trajectory");
        }
    }
}
