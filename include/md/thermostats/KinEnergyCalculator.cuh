#ifndef KIN_ENERGY_CALCULATOR_CUH
#define KIN_ENERGY_CALCULATOR_CUH

#include <md/core/State.cuh>

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

#endif