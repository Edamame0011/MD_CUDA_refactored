#ifndef BUSSI_THERMOSTAT_CUH
#define BUSSI_THERMOSTAT_CUH

#include <md/thermostats/Thermostat.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/core/State.cuh>
#include <curand.h>
#include <cuda_runtime.h>

namespace md::thermostats {
    class BussiThermostat : public Thermostat {
        public: 
            BussiThermostat(const float _tau, int seed, TemperatureScheduler *_scheduler) : tau(_tau), scheduler(_scheduler) {
                // 乱数生成器の初期化
                curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
                curandSetPseudoRandomGeneratorSeed(gen, seed);
            }

            ~BussiThermostat() {
                curandDestroyGenerator(gen);
            }

            void stepOne(State& state) override { /*何もしない*/ }
            void stepTwo(State& state) override;

            void init(const State& state);
        private:
            void generate_rand(const State& state);

            float tau;
            int dof;
            thrust::device_vector<float> random;
            TemperatureScheduler *scheduler;
            curandGenerator_t gen;
    };
}

#endif