#include <md/observers/LinearExportTrajectory.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <iomanip>

using namespace md::observers;

void LinearExportTrajectory::output(State& state, Interaction* interaction) {
    if (state.current_steps % this->output_interval == 0) {
        print_energies(state, interaction);
    }
}

void LinearExportTrajectory::init(State& state, Interaction* interaction) {
    std::cout << "time, kinetic energy, potential energy, total energy, temperature" << std::endl;
    print_energies(state, interaction);
}
