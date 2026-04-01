#ifndef NOSE_HOOVER_CHAIN_CUH
#define NOSE_HOOVER_CHAIN_CUH

#include <md/thermostats/Thermostat.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/thermostats/KinEnergyCalculator.cuh>
#include <md/core/State.cuh>

struct ChainState {
    float *pos, *vel, *force;
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
            float c_mass = 0.0f;
            ChainState c_state;
            TemperatureScheduler *scheduler = nullptr;
            std::unique_ptr<KinEnergyCalculator> calculator = nullptr;
    };
}

#endif