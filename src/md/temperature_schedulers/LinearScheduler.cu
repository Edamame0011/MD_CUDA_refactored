#include <md/temperature_schedulers/LinearScheduler.hpp>

#include <md/core/State.hpp>

namespace md::temperature_schedulers {
    void LinearScheduler::get_temperature(State& state, SimState& simstate) {
        float targ_temp = initial_temperature + (rate_per_step * simstate.current_steps);
        cudaMemcpyToSymbolAsync(c_target_temperature, &targ_temp, sizeof(float), 0, cudaMemcpyHostToDevice, simstate.stream);
    }
}