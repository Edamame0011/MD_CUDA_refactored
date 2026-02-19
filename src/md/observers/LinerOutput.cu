#include <md/observers/LinearOutput.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <iomanip>

using namespace md::observers;

void LinearOutput::output(const State& state, int step) {
    if (step % this->output_interval == 0) {
        float K = md::utils::compute::calc_kinetic_energy(state);
        float U = state.potential_energy;

        int dof = 3 * state.n_atoms;
        float temperature = 2 * K / (dof * boltzmann_constant);

        std::cout << std::setprecision(15) << std::scientific << step * state.dt << ", "
                                                              << K << ", "
                                                              << U << ", "
                                                              << K + U << ", "
                                                              << temperature << std::endl;
    }
}