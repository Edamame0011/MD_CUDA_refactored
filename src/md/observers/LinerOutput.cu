#include <md/observers/LinearOutput.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <iomanip>

using namespace md::observers;

void LinearOutput::output(State& state, Interaction* interaction) {
    if (state.current_steps % this->output_interval == 0) {
        auto view = state.get_view();
        auto pos = view.pos;
        std::cout << "[[Debug]]" << pos.x << " " << pos.y << " " << pos.z << std::endl;
        print_energies(state, interaction);
    }
}

void LinearOutput::init(State& state, Interaction* interaction) {
    std::cout << "time, kinetic energy, potential energy, total energy, temperature" << std::endl;
    print_energies(state, interaction);
}
