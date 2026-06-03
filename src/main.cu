#include <md/cells/Cell.cuh>
#include <md/cells/CubicCell.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/integrators/ConstantVolume.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/interactions/NNP_aoti.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/observers/Observer.cuh>
#include <md/observers/LogOutput.cuh>
#include <md/thermostats/Thermostat.cuh>
#include <md/thermostats/NoThermostat.cuh>
#include <md/utils/initialize.cuh>
#include <md/core/Simulator.cuh>

#include <string>

int main() {
    std::mt19937 mt(12345);
    std::string model_path = "/home/nozawa/SchNet/model_schnet_aoti.pt2";
    std::string input_path = "./data/sample_NS2.xyz";

    std::array<std::array<float, 3>, 3> lattice;
    std::unique_ptr<md::State> state = nullptr;
    std::unique_ptr<md::Integrator> integrator = nullptr;
    std::unique_ptr<md::Interaction> interaction = nullptr;
    std::unique_ptr<md::Observer> observer = nullptr;
    std::unique_ptr<md::Thermostat> thermostat = nullptr;
    std::unique_ptr<md::Cell> cell = nullptr;
    std::unique_ptr<md::NeighbourList> nl = nullptr;

    state = md::utils::initialize::read_state_from_xyz(lattice, input_path);
    nl = std::make_unique<md::NeighbourList>(*(state.get()), 5.0f, 1.0f);
    cell = std::make_unique<md::cells::CubicCell>(lattice);
    md::utils::initialize::init_velocities(*(state.get()), 300, mt);
    thermostat = std::make_unique<md::thermostats::NoThermostat>();
    integrator = std::make_unique<md::integrators::ConstantVolume>(thermostat.get());
    interaction = std::make_unique<md::interactions::NNP_aoti>(*(state.get()), cell.get(), nl.get(), 5.0f, 70000, model_path);
    observer = std::make_unique<md::observers::LogOutput>(9, 5, interaction.get());

    md::Simulator simulator(*(state.get()), interaction.get(), integrator.get(), observer.get(), cell.get());
    simulator.run(1e+5, false);
}