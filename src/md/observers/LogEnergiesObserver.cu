#include <md/observers/LogEnergiesObserver.hpp>

#include <md/core/constant.h>
#include <md/core/State.hpp>
#include <md/observers/EnergiesPrinter.hpp>

#include <cmath>

namespace md::observers {
    LogEnergiesObserver::LogEnergiesObserver(float _interval, int _counter, Interaction* _interaction, const std::string& output_path)
    : log_interval(_interval), counter(_counter), printer(std::make_unique<EnergiesPrinter>(_interaction, output_path)) {
            this->checker = 1e-3 * std::pow(log_interval, counter);
        }

    LogEnergiesObserver::~LogEnergiesObserver() = default;

    void LogEnergiesObserver::output(State& state, SimState& simstate) {
        if (simstate.dt * simstate.current_steps > checker) {
            printer->print_energies(state, simstate);

            this->counter ++;
            this->checker = 1e-3 * std::pow(log_interval, counter);
        }
    }

    void LogEnergiesObserver::init(State& state, SimState& simstate) {
        printer->print_energies(state, simstate);
    }
}

