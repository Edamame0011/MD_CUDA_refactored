#pragma once

#include <md/integrators/Integrator.cuh>
#include <thrust/device_vector.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace md {
    class State;
    class TemperatureScheduler;
}

namespace md::integrators {
    class LangevinIntegrator : public Integrator {
        public: 
            LangevinIntegrator(float _gamma, int seed, TemperatureScheduler* _scheduler) : gamma(_gamma), scheduler(_scheduler) {}

            ~LangevinIntegrator() {
            }

            void integrateStepOne(State& state) override;
            void integrateStepTwo(State& state) override;
        
            void init(const State& satate, unsigned long long seed);
            
        private:
            float gamma;
            float c1;
            int dof;
            TemperatureScheduler* scheduler = nullptr;
            thrust::device_vector<curandState> curand_state;
    };
}