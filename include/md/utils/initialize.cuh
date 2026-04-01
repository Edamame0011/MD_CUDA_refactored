#ifndef INITIALIZE_CUH
#define INITIALIZE_CUH

#include <md/core/State.cuh>
#include <string>
#include <random>
#include <array>
#include <random>

#include <md/core/Simulator.cuh>

namespace md::utils::initialize {
    void init_velocities(State& state, float temperature, std::mt19937& mt);
    std::unique_ptr<md::State> generate_binary_lj(const int n_atoms, const float density, std::array<std::array<float, 3>, 3>& lattice, const float a_ratio, std::mt19937 &mt);
}

#endif