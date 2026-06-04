#pragma once

namespace md {
    class State;
}

namespace md::thermostats {
    class KinEnergyCalculator {
    public:
        KinEnergyCalculator(State& state);
        ~KinEnergyCalculator();
        void calc_kinetic_energy(State& state);    

    private:
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
    };
}