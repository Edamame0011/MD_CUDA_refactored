#include <md/observers/EnergiesPrinter.cuh>

#include <md/core/State.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>

#include <iomanip>

namespace md::observers {
    EnergiesPrinter::EnergiesPrinter(Interaction* _interaction, const std::string& output_path): interaction(_interaction) {
        this->ofs.open(output_path);
        if (!ofs) {
            throw std::runtime_error("出力ファイルが開けませんでした。");
        }

        ofs << "time, kinetic energy, potential energy, total energy, temperature" << std::endl;
    }
    void EnergiesPrinter::print_energies(State& state) {
        interaction->calc_potential(state);
        float K = md::utils::compute::calc_kinetic_energy(state);
        float U = state.potential_energy;

        int dof = 3 * state.n_atoms;
        float temperature = 2 * K / (dof * boltzmann_constant);

        ofs << std::setprecision(7) << std::scientific << state.current_steps * state.dt << ", "
                                                              << K << ", "
                                                              << U << ", "
                                                              << K + U << ", "
                                                              << temperature << std::endl;
    }
}