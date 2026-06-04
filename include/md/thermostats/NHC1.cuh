#pragma once

#include <md/thermostats/Thermostat.cuh>

#include <memory>

namespace md {
    class State;
    class TemperatureScheduler;

    namespace thermostats {
        class KinEnergyCalculator;
    }
}

struct ChainState {
    float *pos, *vel, *force, *mass;
    float *scaling_factor;
};

namespace md::thermostats {
    class NHC1 : public Thermostat {
        public: 
            NHC1(const float _tau, TemperatureScheduler *_scheduler);
            ~NHC1();
            void stepOne(State& state) override;
            void stepTwo(State& state) override;

            void init(State& state);
        
        private:
            void op(State& state);
            
            float tau = 0.0f; 
            float dof = 0.0f;
            ChainState c_state;
            TemperatureScheduler *scheduler = nullptr;
            std::unique_ptr<KinEnergyCalculator> calculator;
    };
}