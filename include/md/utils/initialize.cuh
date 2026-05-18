#ifndef INITIALIZE_CUH
#define INITIALIZE_CUH

#include <md/core/State.cuh>
#include <string>
#include <random>
#include <array>
#include <random>

#include <md/core/Simulator.cuh>

namespace md::utils::initialize {
    std::unique_ptr<md::State> read_state_from_xyz(std::array<std::array<float, 3>, 3>& lattice, const std::string& path);
    std::array<std::array<float, 3>, 3> find_lattice_from_xyz(const std::string& path);
    void init_velocities(State& state, float temperature, std::mt19937& mt);
    std::unique_ptr<md::State> generate_binary_lj(const int n_atoms, const float density, std::array<std::array<float, 3>, 3>& lattice, const float a_ratio, std::mt19937 &mt);
}

#endif