#ifndef MD_THERMOSTAT_CUH
#define MD_THERMOSTAT_CUH

#include <md/core/State.cuh>

namespace md {
    class Thermostat {
        public: 
            virtual ~Thermostat() = default;

            virtual void stepOne(State& state) = 0;
            virtual void stepTwo(State& state) = 0;
        
        protected:
            Thermostat() = default;
    };
}

#endif