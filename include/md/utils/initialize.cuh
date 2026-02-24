#ifndef INITIALIZE_CUH
#define INITIALIZE_CUH

#include <md/core/State.cuh>
#include <string>
#include <random>
#include <array>

namespace md::utils::initialize {
    void read_state_from_xyz(State& state, const std::string& path);
    std::array<std::array<float, 3>, 3> find_lattice_from_xyz(const std::string& path);
    void init_velocities(State& state, float temperature, std::mt19937& mt);
}

#endif