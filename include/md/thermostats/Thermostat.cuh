#pragma once

namespace md {
    class State;
    class SimState;

    extern __constant__ float c_target_temperature;

    class Thermostat {
        public: 
            virtual ~Thermostat() = default;

            virtual void stepOne(State& state, SimState& simstate) = 0;
            virtual void stepTwo(State& state, SimState& simstate) = 0;
        
        protected:
            Thermostat() = default;
    };
}