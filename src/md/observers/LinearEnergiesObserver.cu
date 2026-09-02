#include <md/observers/LinearEnergiesObserver.hpp>

#include <md/core/constant.h>
#include <md/core/State.hpp>
#include <md/observers/EnergiesPrinter.hpp>

#include <iomanip>

using namespace md::observers;

LinearEnergiesObserver::LinearEnergiesObserver(int interval, Interaction* _interaction, const std::string& output_path)
: output_interval(interval), printer(std::make_unique<EnergiesPrinter>(_interaction, output_path)) {}

LinearEnergiesObserver::~LinearEnergiesObserver() = default;

void LinearEnergiesObserver::output(State& state, SimState& simstate) {
    if (simstate.current_steps % this->output_interval == 0) {
        printer->print_energies(state, simstate);
    }
}

void LinearEnergiesObserver::init(State& state, SimState& simstate) {
    printer->print_energies(state, simstate);
}
