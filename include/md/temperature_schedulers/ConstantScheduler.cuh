#ifndef CONSTANT_SCHEDULER_CUH
#define CONSTANT_SCHEDULER_CUH

#include <md/temperature_schedulers/TemperatureScheduler.cuh>

namespace md::temperature_schedulers {
    class ConstantScheduler : public TemperatureScheduler {
        public: 
            ConstantScheduler(float _target_temperature): target_temperature(_target_temperature) {
                cudaMemcpyToSymbol(c_target_temperature, &this->target_temperature, sizeof(float), 0, cudaMemcpyHostToDevice);
            }
            void get_temperature(State& state) override { 
                // 何もしない
            }
        private:
            float target_temperature;
    };
}

#endif