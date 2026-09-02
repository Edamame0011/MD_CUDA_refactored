#include <md/thermostats/NHC1.hpp>

#include <md/core/State.hpp>
#include <md/core/constant.h>
#include <md/temperature_schedulers/TemperatureScheduler.hpp>
#include <md/thermostats/KinEnergyCalculator.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace {
    __global__ void update_mass (
        md::thermostats::NHC1ChainState* chain, 
        float tau, 
        float boltzmann_constant, 
        float dof
    ) {
        chain->mass = tau * tau * boltzmann_constant * md::c_target_temperature * dof;
    }

    __global__ void calc_scaling_factor(
        const float* kinetic_energy,
        md::thermostats::NHC1ChainState* chain,
        float dt,
        float tau,
        int degrees_of_freedom,
        float boltzmann_constant
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            auto AKIN = 2.0f * *kinetic_energy;
            auto dt_half = 0.5f * dt;
            auto dt_quarter = 0.25f * dt;

            auto f0 = chain->force;
            auto v0 = chain->velocity;
            auto p0 = chain->position;
            auto m0_inv = 1.0f / chain->mass;

            auto targ_kin = md::c_target_temperature * degrees_of_freedom * boltzmann_constant;

            // 逆順の更新
            f0 = (AKIN - targ_kin) * m0_inv;
            v0 += dt_quarter * f0;

            // スケーリング
            float sf = __expf(-dt_half * v0);
            AKIN *= __expf(-dt * v0);

            // 変位の更新
            p0 += dt_half * v0;

            // 順方向の更新
            f0 = (AKIN - targ_kin) * m0_inv;
            v0 += dt_quarter * f0;

            // グローバルメモリに書き込み
            chain->force = f0;
            chain->velocity = v0;
            chain->position = p0;
            chain->scaling_factor = sf;
        }
    }

    __global__ void scale_velocities(
        md::DeviceVec3 velocity,
        int atom_count,
        const md::thermostats::NHC1ChainState* chain
    ) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= atom_count) return;

        const float scaling = chain->scaling_factor;
        velocity.x[index] *= scaling;
        velocity.y[index] *= scaling;
        velocity.z[index] *= scaling;
    }
}

namespace md::thermostats {
    NHC1::NHC1(float tau, TemperatureScheduler* scheduler)
        : tau_(tau), scheduler_(scheduler) {
        if (!std::isfinite(tau_) || tau_ <= 0.0f) {
            throw std::invalid_argument("NHC1 tau must be finite and positive");
        }
        if (scheduler_ == nullptr) {
            throw std::invalid_argument("NHC1 requires a temperature scheduler");
        }
        cudaMalloc(&chain_state_, sizeof(NHC1ChainState));
    }

    NHC1::~NHC1() {
        cudaFree(chain_state_);
    }

    void NHC1::init(State& state, SimState& simstate) {
        if (state.n_atoms <= 0 || state.n_atoms > (2147483647 / 3)) {
            throw std::invalid_argument("invalid atom count for NHC1");
        }
        if (!std::isfinite(boltzmann_constant) || boltzmann_constant <= 0.0f) {
            throw std::invalid_argument("boltzmann_constant must be finite and positive");
        }

        atom_count_ = state.n_atoms;
        degrees_of_freedom_ = 3 * atom_count_;
        calculator_ = std::make_unique<KinEnergyCalculator>(state);
        cudaMemsetAsync(chain_state_, 0, sizeof(*chain_state_), simstate.stream);
    }

    void NHC1::stepOne(State& state, SimState& simstate) {
        apply(state, simstate);
    }

    void NHC1::stepTwo(State& state, SimState& simstate) {
        apply(state, simstate);
    }

    void NHC1::apply(State& state, SimState& simstate) {
        scheduler_->get_temperature(state, simstate);

        update_mass<<<1, 1, 0, simstate.stream>>>(
            chain_state_, 
            tau_, 
            boltzmann_constant, 
            degrees_of_freedom_
        );

        calculator_->calc_kinetic_energy(state, simstate);

        calc_scaling_factor<<<1, 1, 0, simstate.stream>>>(
            calculator_->device_kinetic_energy(),
            chain_state_,
            simstate.dt,
            tau_,
            degrees_of_freedom_,
            boltzmann_constant
        );

        const int blocks = (atom_count_ + NUM_THREADS - 1) / NUM_THREADS;
        scale_velocities<<<blocks, NUM_THREADS, 0, simstate.stream>>>(
            state.vel, 
            atom_count_, 
            chain_state_
        );
    }
}
