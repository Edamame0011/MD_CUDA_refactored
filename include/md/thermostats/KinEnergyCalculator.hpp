#pragma once

#include <cstddef>

namespace md {
    class State;
    class SimState;
}

namespace md::thermostats {
    class KinEnergyCalculator final {
        public:
            explicit KinEnergyCalculator(const State& state);
            ~KinEnergyCalculator();

            void calc_kinetic_energy(const State& state, SimState& simstate);
            float calc_kinetic_energy_host(const State& state, SimState& simstate);

            float* device_kinetic_energy() noexcept { return d_kinetic_energy_; }
            const float* device_kinetic_energy() const noexcept { return d_kinetic_energy_; }

            KinEnergyCalculator(const KinEnergyCalculator&) = delete;
            KinEnergyCalculator& operator=(const KinEnergyCalculator&) = delete;

        private:
            int capacity_ = 0;
            void* d_temp_storage_ = nullptr;
            std::size_t temp_storage_bytes_ = 0;
            float* d_kinetic_energy_ = nullptr;
    };
}
