#pragma once

#include <md/temperature_schedulers/TemperatureScheduler.hpp>

namespace md::temperature_schedulers {
    class ConstantScheduler final : public TemperatureScheduler {
        public:
            explicit ConstantScheduler(float target_temperature);

            void get_temperature(State& state, SimState& simstate) override;

            float target_temperature() const noexcept { return target_temperature_; }

        private:
            float target_temperature_;
    };
}
