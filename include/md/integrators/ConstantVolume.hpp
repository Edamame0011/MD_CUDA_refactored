#pragma once

#include <md/integrators/Integrator.hpp>

namespace md {
    class State;
    class SimState;
    class Thermostat;

    namespace integrators {
        class ConstantVolume : public Integrator {
            public: 
                ConstantVolume(Thermostat* thermostat_) : thermostat(thermostat_) {}

                void integrateStepOne(State* state, SimState& simstate) override;
                void integrateStepTwo(State* state, SimState& simstate) override;

                ConstantVolume(const ConstantVolume&) = delete;
                ConstantVolume& operator=(const ConstantVolume&) = delete;
            private:
                Thermostat* thermostat;
        };
    }
}