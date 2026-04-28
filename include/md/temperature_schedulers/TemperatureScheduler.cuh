#ifndef TEMPERATURE_SCHEDULER_CUH
#define TEMPERATURE_SCHEDULER_CUH

#include <md/core/State.cuh>

namespace md {
    class TemperatureScheduler {
        public: 
            virtual ~TemperatureScheduler() = default;
            
            virtual void get_temperature(State& state) = 0;

        protected:
            TemperatureScheduler() = default;
    };
}

#endif