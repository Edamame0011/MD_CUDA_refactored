#pragma once

namespace md {
    class State;
}

namespace md::utils::compute {
    float calc_kinetic_energy(State& state);
    void remove_drift(State& state);
}