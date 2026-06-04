#pragma once

#include <md/temperature_schedulers/TemperatureScheduler.cuh>

namespace md::temperature_schedulers {
    class LinearScheduler : public TemperatureScheduler {
        public: 
            LinearScheduler(float _initial_temperature, float _rate_per_step): initial_temperature(_initial_temperature), rate_per_step(_rate_per_step) {}
            void get_temperature(State& state) override {
                float targ_temp = initial_temperature + (rate_per_step * state.current_steps);
                cudaMemcpyToSymbolAsync(c_target_temperature, &targ_temp, sizeof(float), 0, cudaMemcpyHostToDevice, state.stream);
            }
        private:
            float initial_temperature;
            float rate_per_step;
    };
}