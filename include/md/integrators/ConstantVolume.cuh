#ifndef CONSTANT_VOLUME_CUH
#define CONSTANT_VOLUME_CUH

#include <md/integrators/Integrator.cuh>
#include <md/thermostats/Thermostat.cuh>
#include <md/core/State.cuh>

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

#endif