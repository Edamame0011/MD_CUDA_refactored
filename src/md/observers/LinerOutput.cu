#include <md/observers/LinearOutput.cuh>

#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/observers/EnergiesPrinter.cuh>

#include <iomanip>

using namespace md::observers;

LinearOutput::LinearOutput(int interval, Interaction* _interaction, const std::string& output_path)
: output_interval(interval), printer(std::make_unique<EnergiesPrinter>(_interaction, output_path)) {}

LinearOutput::~LinearOutput() = default;

void LinearOutput::output(State& state) {
    if (state.current_steps % this->output_interval == 0) {
        printer->print_energies(state);
    }
}

void LinearOutput::init(State& state) {
    printer->print_energies(state);
}
