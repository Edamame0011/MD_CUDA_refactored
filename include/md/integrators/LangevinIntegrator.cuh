#pragma once

#include <md/integrators/Integrator.hpp>

#include <curand_kernel.h>
#include <thrust/device_vector.h>

namespace md {
    class TemperatureScheduler;

    namespace integrators {
        class LangevinIntegrator : public Integrator {
            public:
                LangevinIntegrator(
                    float gamma, unsigned long long seed, TemperatureScheduler* scheduler);

                void init(const State* state, SimState& simstate);
                void init(const State* state, SimState& simstate, unsigned long long seed);

                void integrateStepOne(State* state, SimState& simstate) override;
                void integrateStepTwo(State* state, SimState& simstate) override;

                LangevinIntegrator(const LangevinIntegrator&) = delete;
                LangevinIntegrator& operator=(const LangevinIntegrator&) = delete;

            private:
                float gamma_;
                unsigned long long seed_;
                float dof_;
                float c1_;
                int atom_count_ = 0;
                TemperatureScheduler* scheduler_ = nullptr;
                thrust::device_vector<curandState> curand_state_;
        };
    }
}