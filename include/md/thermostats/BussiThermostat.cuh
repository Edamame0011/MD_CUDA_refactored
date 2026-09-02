#pragma once

#include <md/thermostats/Thermostat.cuh>

#include <curand_kernel.h>
#include <memory>

namespace md {
    class TemperatureScheduler;

    namespace thermostats {
        class KinEnergyCalculator;

        class BussiThermostat : public Thermostat {
            public:
                BussiThermostat(float tau, TemperatureScheduler* scheduler);
                ~BussiThermostat();

                void init(State& state, SimState& simstate, unsigned long long seed);
                void stepOne(State& state, SimState& simstate) override;
                void stepTwo(State& state, SimState& simstate) override;

                BussiThermostat(const BussiThermostat&) = delete;
                BussiThermostat& operator=(const BussiThermostat&) = delete;

            private:
                float tau_;
                int degrees_of_freedom_ = 0;
                int atom_count_ = 0;
                TemperatureScheduler* scheduler_ = nullptr;
                std::unique_ptr<KinEnergyCalculator> calculator_;
                float* scaling_factor_ = nullptr;

                // 乱数生成
                curandState* curand_state_ = nullptr;
        };
    }
}