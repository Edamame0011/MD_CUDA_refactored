#ifndef LINEAR_SCHEDULER_CUH
#define LINEAR_SCHEDULER_CUH

#include <md/temperature_schedulers/TemperatureScheduler.cuh>

namespace md::temperature_schedulers {
    class LinearScheduler : public TemperatureScheduler {
        public: 
            LinearScheduler(float _initial_temperature, float _rate_per_step): initial_temperature(_initial_temperature), rate_per_step(_rate_per_step) {}
            float get_temperature(const int step) override {
                return initial_temperature + (rate_per_step * step);
            }
        private:
            float initial_temperature;
            float rate_per_step;
    };
}

#endif