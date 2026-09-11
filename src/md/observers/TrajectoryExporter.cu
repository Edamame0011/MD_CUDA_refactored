#include <md/observers/TrajectoryExporter.hpp>

#include <md/core/Cell.cuh>
#include <md/core/State.hpp>
#include <md/interactions/Interaction.hpp>

#include <cuda_runtime.h>

#include <iomanip>
#include <stdexcept>

namespace {
    void check_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("TrajectoryExporter: ") + operation + ": " + cudaGetErrorString(status));
        }
    }
}

namespace md::observers {
    TrajectoryExporter::TrajectoryExporter(
        State* state, 
        Cell& cell, 
        Interaction* interaction, 
        const std::string& output_path,
        const std::vector<std::string>& species_to_symbol
    ) : output_(output_path), cell_(cell), interaction_(interaction), species_to_symbol_(species_to_symbol) {
        if (!output_) {
            throw std::runtime_error("TrajectoryExporter: could not open output file: " + output_path);
        }

        const auto N = static_cast<std::size_t>(state->n_atoms);
        positions_.resize(3 * N);
        forces_.resize(3 * N);
        images_.resize(3 * N);
        species_.resize(N);
        particle_ids_.resize(N);
    }

    void TrajectoryExporter::export_frame(State* state, SimState& simstate, bool unwrap) {
        const auto N = static_cast<std::size_t>(state->n_atoms);

        check_cuda(cudaMemcpy(positions_.data(), state->pos.x, N * sizeof(float), cudaMemcpyDeviceToHost), "copy x positions");
        check_cuda(cudaMemcpy(positions_.data() + N, state->pos.y, N * sizeof(float), cudaMemcpyDeviceToHost), "copy y positions");
        check_cuda(cudaMemcpy(positions_.data() + 2 * N, state->pos.z, N * sizeof(float), cudaMemcpyDeviceToHost), "copy z positions");
        check_cuda(cudaMemcpy(forces_.data(), state->force.x, N * sizeof(float), cudaMemcpyDeviceToHost), "copy x forces");
        check_cuda(cudaMemcpy(forces_.data() + N, state->force.y, N * sizeof(float), cudaMemcpyDeviceToHost), "copy y forces");
        check_cuda(cudaMemcpy(forces_.data() + 2 * N, state->force.z, N * sizeof(float), cudaMemcpyDeviceToHost), "copy z forces");
        check_cuda(cudaMemcpy(species_.data(), state->species, N * sizeof(int), cudaMemcpyDeviceToHost), "copy species");
        check_cuda(cudaMemcpy(particle_ids_.data(), state->particle_id, N * sizeof(int), cudaMemcpyDeviceToHost), "copy particle IDs");
        if (unwrap) {
            check_cuda(cudaMemcpy(images_.data(), state->image.x, N * sizeof(int), cudaMemcpyDeviceToHost), "copy x images");
            check_cuda(cudaMemcpy(images_.data() + N, state->image.y, N * sizeof(int), cudaMemcpyDeviceToHost), "copy y images");
            check_cuda(cudaMemcpy(images_.data() + 2 * N, state->image.z, N * sizeof(int), cudaMemcpyDeviceToHost), "copy z images");
        }

        std::vector<int> current_index_by_id(N, -1);
        for (int i = 0; i < N; i ++) {
            const int id = particle_ids_[i];
            current_index_by_id[id] = i;
        }

        const auto lattice = cell_.get_lattice();
        const float potential = interaction_->calc_potential(state, simstate);
        output_ << std::scientific << std::setprecision(7);
        output_ << N << '\n';
        output_ << "Lattice=\"" << lattice[0] << " 0.0 0.0 0.0 "
                << lattice[1] << " 0.0 0.0 0.0 " << lattice[2]
                << "\" Properties=species:S:1:pos:R:3:forces:R:3 energy=" << potential << " pbc=\"T T T\"\n";

        for (std::size_t id = 0; id < N; ++id) {
            const auto current = (std::size_t)current_index_by_id[id];
            const float x = positions_[current] + (unwrap ? images_[current] * lattice[0] : 0.0f);
            const float y = positions_[N + current] + (unwrap ? images_[N + current] * lattice[1] : 0.0f);
            const float z = positions_[2 * N + current] + (unwrap ? images_[2 * N + current] * lattice[2] : 0.0f);
            output_ << species_to_symbol_[species_[current]] << ' '
                    << x << ' ' << y << ' ' << z << ' '
                    << forces_[current] << ' ' << forces_[N + current] << ' ' << forces_[2 * N + current] << '\n';
        }
        output_.flush();
        if (!output_) {
            throw std::runtime_error("TrajectoryExporter: failed while writing trajectory");
        }
    }
}
