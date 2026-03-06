#ifndef LANGEVIN_INTEGRATOR_CUH
#define LANGEVIN_INTEGRATOR_CUH

#include <md/integrators/Integrator.cuh>
#include <md/core/State.cuh>
#include <thrust/device_vector.h>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <curand.h>
#include <cuda_runtime.h>

namespace md::integrators {
    class LangevinIntegrator : public Integrator {
        public: 
            LangevinIntegrator(float _gamma, int seed, TemperatureScheduler* _scheduler) : gamma(_gamma), scheduler(_scheduler) {
                // 乱数生成器の初期化
                curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
                curandSetPseudoRandomGeneratorSeed(gen, seed);
            }

            ~LangevinIntegrator() {
                curandDestroyGenerator(gen);
            }

            void integrateStepOne(State& state) override;
            void integrateStepTwo(State& state) override;
        
            void init(const State& satate);
            
        private:
            void init_random(const State& state);
            float gamma;
            float c1;
            int dof;
            struct device_float3 {
                thrust::device_vector<float> x, y, z;
            } random;
            TemperatureScheduler *scheduler;
            curandGenerator_t gen;
    };
}

#endif