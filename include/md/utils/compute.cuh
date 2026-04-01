#ifndef VELOCITY_MANAGER_CUH
#define VELOCITY_MANAGER_CUH

#include <md/core/State.cuh>

namespace md::utils::compute {
    float calc_kinetic_energy(State& state);
    void remove_drift(State& state);
}

#endif