#include <md/observers/Observer.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <iomanip>

void md::observers::print_energies(State& state, Interaction* interaction) {
        interaction->calc_potential(state);
        float K = md::utils::compute::calc_kinetic_energy(state);
        float U = state.potential_energy;

        int dof = 3 * state.n_atoms;
        float temperature = 2 * K / (dof * boltzmann_constant);

        std::cout << std::setprecision(7) << std::scientific << state.current_steps * state.dt << ", "
                                                              << K << ", "
                                                              << U << ", "
                                                              << K + U << ", "
                                                              << temperature << std::endl;
}