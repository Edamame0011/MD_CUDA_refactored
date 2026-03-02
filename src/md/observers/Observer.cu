#include <md/observers/Observer.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <iomanip>

void md::observers::print_energies(const State& state, const int step) {
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