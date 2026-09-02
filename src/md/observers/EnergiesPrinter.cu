#include <md/observers/EnergiesPrinter.hpp>

#include <md/core/State.hpp>
#include <md/interactions/Interaction.hpp>
#include <md/core/constant.h>

#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <iomanip>

using DeviceVec3 = md::DeviceVec3;

namespace {
    struct CalcKinEnergy {
        DeviceVec3 vel;
        float* __restrict__ mass;

        CalcKinEnergy(
            DeviceVec3 vel_, 
            float* mass_
        ) : vel(vel_), mass(mass_) {}

        __host__ __device__ float operator() (int idx) {
            auto vx = vel.x[idx];
            auto vy = vel.y[idx];
            auto vz = vel.z[idx];

            return 0.5 * mass[idx] * (vx * vx + vy * vy + vz * vz);
        }
    };
}

namespace md::observers {
    EnergiesPrinter::EnergiesPrinter(Interaction* _interaction, const std::string& output_path): interaction(_interaction) {
        this->ofs.open(output_path);
        if (!ofs) {
            throw std::runtime_error("出力ファイルが開けませんでした。");
        }

        ofs << "time, kinetic energy, potential energy, total energy, temperature" << std::endl;
    }
    void EnergiesPrinter::print_energies(State& state, SimState& simstate) {
        auto N = state.n_atoms;

        float kinetic_energy = thrust::transform_reduce(
            thrust::device, 
            thrust::make_counting_iterator<int>(0), 
            thrust::make_counting_iterator<int>(N),  
            CalcKinEnergy(
                state.vel, 
                state.mass
            ), 
            0.0f, 
            thrust::plus<float>()
        );
        float potential_energy = interaction->calc_potential(state, simstate);

        int dof = 3 * state.n_atoms;
        float temperature = 2 * kinetic_energy / (dof * boltzmann_constant);

        ofs << std::setprecision(7) << std::scientific << simstate.current_steps * simstate.dt << ", "
                                                              << kinetic_energy << ", "
                                                              << potential_energy << ", "
                                                              << kinetic_energy + potential_energy << ", "
                                                              << temperature << std::endl;
    }
}