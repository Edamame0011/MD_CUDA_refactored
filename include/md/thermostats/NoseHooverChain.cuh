#ifndef NOSE_HOOVER_CHAIN_CUH
#define NOSE_HOOVER_CHAIN_CUH

#include <md/thermostats/Thermostat.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/core/State.cuh>

namespace md::thermostats {
    class NoseHooverChain : public Thermostat {
        public: 
            NoseHooverChain(const int _length, const float _tau, TemperatureScheduler *_scheduler);
            void stepOne(State& state) override;
            void stepTwo(State& state) override;

            void init(State& state);
        
        private:
            std::function<void(State& state)> update_step_one;
            std::function<void(State& state)> update_step_two;

            void NHC1(State& state);
            void NHC2(State& state);
            void NHCM(State& state);

            void set_masses(float target_temperature);
            
            int length;
            std::vector<float> chain_positions, chain_masses, chain_velocities, chain_forces;
            float tau, dof;
            TemperatureScheduler *scheduler;
    };
}

#endif