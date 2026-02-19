#ifndef COMPUTE_CUH
#define COMPUTE_CUH

#include <md/core/State.cuh>

namespace md::utils::compute {
    float calc_kinetic_energy(const State& state);
    void remove_drift(State& state);
}

#endif