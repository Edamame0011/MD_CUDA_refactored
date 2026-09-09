#pragma once

#include <md/temperature_schedulers/TemperatureScheduler.hpp>

namespace md::temperature_schedulers {
    class LinearScheduler : public TemperatureScheduler {
        public: 
            LinearScheduler(float _initial_temperature, float _rate_per_step): initial_temperature(_initial_temperature), rate_per_step(_rate_per_step) {}
            void get_temperature(State* state, SimState& simstate) override;
        private:
            float initial_temperature;
            float rate_per_step;
    };
}