#include <md/core/Simulator.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/core/Simulator.cuh>
#include <md/integrators/ConstantVolume.cuh>
#include <md/interactions/LJPotential.cuh>
#include <md/observers/LinearOutput.cuh>
#include <md/thermostats/NoThermostat.cuh>
#include <md/cells/CubicCell.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/utils/initialize.cuh>
#include <md/observers/LogOutput.cuh>
#include <chrono>

int main() {
    constexpr int n_atoms = 10000;
    constexpr float density = 1.2;
    constexpr float a_ratio = 0.8;
    constexpr float temperature = 1.0;
    constexpr float dt = 1e-4;
    constexpr int seed = 12345;
    constexpr float tsim = 1e+1;

    constexpr float nl_cutoff = 2.0;
    constexpr float nl_margin = 1.0;

    // unitの初期化
    md::conversion_factor = 1.0;
    md::boltzmann_constant = 1.0;

    // stateの初期化
    std::mt19937 mt(seed);
    std::array<std::array<float, 3>, 3> lattice;
    auto state = md::utils::initialize::generate_binary_lj(n_atoms, density, lattice, a_ratio, mt);
    md::utils::initialize::init_velocities(*state, temperature, mt);
    state->dt = dt;

    // Thermostatの初期化
    std::unique_ptr<md::Thermostat> thermostat;
    thermostat = std::make_unique<md::thermostats::NoThermostat>();

    // Integratorの初期化
    std::unique_ptr<md::Integrator> integrator;
    integrator = std::make_unique<md::integrators::ConstantVolume>(thermostat.get());

    // Cellの初期化
    md::cells::CubicCell cell(lattice);

    // NeighbourListの初期化
    md::utils::NeighbourList<md::cells::CubicCell> nl(*state, nl_cutoff, nl_margin);

    // Interactionの初期化
    std::vector<int> identifier;
    std::vector<int> numbers = {0, 1};
    std::vector<int> atomic_numbers(n_atoms);
    cudaMemcpy(atomic_numbers.data(), state->get_view().atomic_numbers, n_atoms * sizeof(int), cudaMemcpyDeviceToHost);
    identifier.reserve(n_atoms);
    int max_val = *std::max_element(numbers.begin(), numbers.end());
    std::vector<int> lut(max_val + 1, -1);
    for (int i = 0; i < 2; ++i) {
        lut[numbers[i]] = i;
    }
    for (int num : atomic_numbers) {
        identifier.push_back(lut[num]);
    }
    md::interactions::LJPotential potential(
        2, 
        cell, 
        &nl, 
        {1.0, 0.8, 0.8, 0.88}, 
        {1.0, 1.5, 1.5, 0.5}, 
        {1.5, 2.0, 2.0, 1.5}, 
        identifier
    );

    // Observerの初期化
    std::unique_ptr<md::Observer> observer;
    observer = std::make_unique<md::observers::LinearOutput>(1000);

    // Simulatorの初期化
    md::Simulator<md::cells::CubicCell> simulator(*state, &potential, integrator.get(), observer.get(), cell);

    auto start = std::chrono::steady_clock::now();
    simulator.run(tsim);
    auto end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;
}