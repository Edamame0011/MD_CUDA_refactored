#ifndef BUSSI_THERMOSTAT_CUH
#define BUSSI_THERMOSTAT_CUH

#include <md/thermostats/Thermostat.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/core/State.cuh>
#include <random>

namespace md::thermostats {
    class BussiThermostat : public Thermostat {
        public: 
            BussiThermostat(const float _tau, int seed, TemperatureScheduler *_scheduler) : tau(_tau), scheduler(_scheduler), gen(seed), normal_dist(0.0f, 1.0f) {}

            void stepOne(State& state) override { /*何もしない*/ }
            void stepTwo(State& state) override;

            void init(const State& state);
        private:
            float tau;
            int dof;
            TemperatureScheduler *scheduler;

            // 乱数生成器
            std::mt19937 gen;
            std::gamma_distribution<float> gamma_dist;
            std::normal_distribution<float> normal_dist;
    };
}

#endif