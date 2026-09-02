#pragma once

#include <md/temperature_schedulers/TemperatureScheduler.hpp>

namespace md::temperature_schedulers {
    class LinearScheduler final : public TemperatureScheduler {
        public:
            LinearScheduler(float initial_temperature, float rate_per_step);

            void get_temperature(State& state, SimState& simstate) override;

            float temperature_at(int step) const noexcept;
            float initial_temperature() const noexcept { return initial_temperature_; }
            float rate_per_step() const noexcept { return rate_per_step_; }

        private:
            float initial_temperature_;
            float rate_per_step_;
            float current_temperature_;
    };
}
