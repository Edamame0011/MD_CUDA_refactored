#pragma once

#include <md/integrators/Integrator.hpp>

namespace md {
    class State;
    class SimState;
    class Thermostat;

    namespace integrators {
        class ConstantVolumeLJ : public Integrator {
            public: 
                ConstantVolumeLJ(Thermostat* thermostat_) : thermostat(thermostat_) {}

                void integrateStepOne(State& state, SimState& simstate) override;
                void integrateStepTwo(State& state, SimState& simstate) override;

            private:
                Thermostat* thermostat;
        };
    }
}