#pragma once

#include <md/temperature_schedulers/TemperatureScheduler.hpp>

namespace md::temperature_schedulers {
    class ConstantScheduler final : public TemperatureScheduler {
        public:
            ConstantScheduler(float target_temperature);

            void get_temperature(State* state, SimState& simstate) override {
                // 何もしない
            }

        private:
            float target_temperature_;
    };
}
