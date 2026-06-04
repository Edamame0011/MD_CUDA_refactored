#pragma once

#include <md/integrators/Integrator.cuh>

namespace md {
    class State;
    class Thermostat;
}

namespace md::integrators {
    class ConstantVolume : public Integrator {
        public: 
            ConstantVolume(Thermostat* _thermostat) : thermostat(_thermostat) {}

            void integrateStepOne(State& state) override;
            void integrateStepTwo(State& state) override;
        
        private:
            Thermostat* thermostat;
    };
}