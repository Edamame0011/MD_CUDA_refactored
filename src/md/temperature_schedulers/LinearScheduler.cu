#include <md/temperature_schedulers/LinearScheduler.hpp>

#include <md/core/State.hpp>
#include <md/thermostats/Thermostat.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace md::temperature_schedulers {
    LinearScheduler::LinearScheduler(float initial_temperature, float rate_per_step)
        : initial_temperature_(initial_temperature),
          rate_per_step_(rate_per_step),
          current_temperature_(initial_temperature) {
        if (!std::isfinite(initial_temperature_) || initial_temperature_ < 0.0f) {
            throw std::invalid_argument("initial temperature must be finite and non-negative");
        }
        if (!std::isfinite(rate_per_step_)) {
            throw std::invalid_argument("temperature rate must be finite");
        }
    }

    float LinearScheduler::temperature_at(int step) const noexcept {
        const float non_negative_step = static_cast<float>(std::max(step, 0));
        const float temperature = std::fma(rate_per_step_, non_negative_step,
                                           initial_temperature_);

        if (!(temperature > 0.0f)) {
            return 0.0f;
        }
        if (!std::isfinite(temperature)) {
            return std::numeric_limits<float>::max();
        }
        return temperature;
    }

    void LinearScheduler::get_temperature(State&, SimState& simstate) {
        current_temperature_ = temperature_at(simstate.current_steps);
        cudaMemcpyToSymbolAsync(
            c_target_temperature,
            &current_temperature_,
            sizeof(current_temperature_),
            0,
            cudaMemcpyHostToDevice,
            simstate.stream
        );
    }
}
