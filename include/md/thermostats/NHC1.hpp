#pragma once

#include <md/thermostats/Thermostat.cuh>

#include <memory>

namespace md {
    class TemperatureScheduler;

    namespace thermostats {
        class KinEnergyCalculator;
    }
}

namespace md::thermostats {
    struct NHC1ChainState {
        float position;
        float velocity;
        float force;
        float mass;
        float scaling_factor;
    };

    class NHC1 final : public Thermostat {
        public:
            NHC1(float tau, TemperatureScheduler* scheduler);
            ~NHC1();

            void init(State& state, SimState& simstate);
            void stepOne(State& state, SimState& simstate) override;
            void stepTwo(State& state, SimState& simstate) override;

            NHC1(const NHC1&) = delete;
            NHC1& operator=(const NHC1&) = delete;

        private:
            void apply(State& state, SimState& simstate);

            float tau_;
            int degrees_of_freedom_ = 0;
            int atom_count_ = 0;
            TemperatureScheduler* scheduler_ = nullptr;
            std::unique_ptr<KinEnergyCalculator> calculator_;
            NHC1ChainState* chain_state_ = nullptr;
    };
}
