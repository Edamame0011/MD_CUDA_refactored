#include <md/temperature_schedulers/ConstantScheduler.hpp>

#include <md/core/State.hpp>
#include <md/thermostats/Thermostat.cuh>

#include <cmath>
#include <stdexcept>

namespace md::temperature_schedulers {
    ConstantScheduler::ConstantScheduler(float target_temperature)
        : target_temperature_(target_temperature) {
            if (!std::isfinite(target_temperature_) || target_temperature_ < 0.0f) {
                throw std::invalid_argument("target temperature must be finite and non-negative");
            }
    }

    void ConstantScheduler::get_temperature(State&, SimState& simstate) {
        cudaMemcpyToSymbolAsync(
            c_target_temperature,
            &target_temperature_,
            sizeof(target_temperature_),
            0,
            cudaMemcpyHostToDevice,
            simstate.stream
        );
    }
}