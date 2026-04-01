#ifndef CONSTANT_SCHEDULER_CUH
#define CONSTANT_SCHEDULER_CUH

#include <md/temperature_schedulers/TemperatureScheduler.cuh>

namespace md::temperature_schedulers {
    class ConstantScheduler : public TemperatureScheduler {
        public: 
            ConstantScheduler(float _target_temperature): target_temperature(_target_temperature) {}
            float get_temperature(const int step) override { 
                return target_temperature; 
            }
        private:
            float target_temperature;
    };
}

#endif