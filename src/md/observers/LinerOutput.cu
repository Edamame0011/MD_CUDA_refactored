#include <md/observers/LinearOutput.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <iomanip>

using namespace md::observers;

void LinearOutput::output(const State& state) {
    if (state.current_steps % this->output_interval == 0) {
        print_energies(state);
    }
}

void LinearOutput::init(const State& state) {
    std::cout << "time, kinetic energy, potential energy, total energy, temperature" << std::endl;
    print_energies(state);
}