#pragma once

namespace md {
    class State;

    
    class TemperatureScheduler {
        public: 
            virtual ~TemperatureScheduler() = default;
            
            virtual void get_temperature(State& state) = 0;

        protected:
            TemperatureScheduler() = default;
    };
}