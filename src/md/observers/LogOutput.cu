#include <md/observers/LogOutput.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <cmath>

using namespace md::observers;

LogOutput::LogOutput(float _interval, int _counter) : log_interval(_interval), counter(_counter) {
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }

void LogOutput::output(State& state, Interaction* interaction) {
    if (state.dt * state.current_steps > checker) {
        print_energies(state, interaction);

        this->counter ++;
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }
}

void LogOutput::init(State& state, Interaction* interaction) {
    std::cout << "time, kinetic energy, potential energy, total energy, temperature" << std::endl;
    print_energies(state, interaction);
}