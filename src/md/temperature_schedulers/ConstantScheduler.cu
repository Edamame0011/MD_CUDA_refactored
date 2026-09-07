#include <md/temperature_schedulers/ConstantScheduler.hpp>

namespace md::temperature_schedulers {
    ConstantScheduler::ConstantScheduler(float target_temperature): target_temperature_(target_temperature) {
        cudaMemcpyToSymbol(c_target_temperature, &this->target_temperature_, sizeof(float), 0, cudaMemcpyHostToDevice);
    }
}