#pragma once

namespace md {
    class State;

    extern __constant__ float c_target_temperature;

    class Thermostat {
        public: 
            virtual ~Thermostat() = default;

            virtual void stepOne(State& state) = 0;
            virtual void stepTwo(State& state) = 0;
        
        protected:
            Thermostat() = default;
    };
}