#include <md/observers/LogOutput.cuh>

#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/observers/EnergiesPrinter.cuh>

#include <cmath>

using namespace md::observers;

LogOutput::LogOutput(float _interval, int _counter, Interaction* _interaction, const std::string& output_path)
: log_interval(_interval), counter(_counter), printer(std::make_unique<EnergiesPrinter>(_interaction, output_path)) {
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }

LogOutput::~LogOutput() = default;

void LogOutput::output(State& state) {
    if (state.dt * state.current_steps > checker) {
        printer->print_energies(state);

        this->counter ++;
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }
}

void LogOutput::init(State& state) {
    printer->print_energies(state);
}
