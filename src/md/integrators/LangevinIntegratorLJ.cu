#include <md/integrators/LangevinIntegratorLJ.cuh>

#include <md/core/State.hpp>
#include <md/core/constant.h>
#include <md/temperature_schedulers/TemperatureScheduler.hpp>
#include <md/thermostats/Thermostat.cuh>

#include <cmath>
#include <stdexcept>
#include <string>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

namespace {
    struct InitCurand {
        curandState* states;
        unsigned long long seed;

        InitCurand(curandState* _states, unsigned long long _seed) 
            : states(_states), seed(_seed) {}

        __device__ void operator() (const int idx) {
            curand_init(seed, idx, 0, &states[idx]);
        }
    };

    __global__ void baoab_step_one(
        md::DeviceVec3 pos,
        md::DeviceVec3 vel,
        md::DeviceVec3 force,
        curandState* random_states,
        const int atom_count,
        const float dt_half,
        const float dt_half_conv,
        const float c1,
        const float boltzmann_constant
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= atom_count) {
            return;
        }

        auto cs = random_states[idx];
        const float c3 = sqrtf(boltzmann_constant * md::c_target_temperature * (1.0f - c1 * c1));

        // 速度の更新1
        auto vx = vel.x[idx] + force.x[idx] * dt_half_conv;
        auto vy = vel.y[idx] + force.y[idx] * dt_half_conv;
        auto vz = vel.z[idx] + force.z[idx] * dt_half_conv;

        // 位置の更新1
        pos.x[idx] += vx * dt_half;
        pos.y[idx] += vy * dt_half;
        pos.z[idx] += vz * dt_half;

        // 乱数の生成
        float rx = curand_normal(&cs);
        float ry = curand_normal(&cs);
        float rz = curand_normal(&cs);

        // 速度の更新2
        vx = c1 * vx + c3 * rx;
        vy = c1 * vy + c3 * ry;
        vz = c1 * vz + c3 * rz;
    
        vel.x[idx] = vx;
        vel.y[idx] = vy;
        vel.z[idx] = vz;

        // 位置の更新2
        pos.x[idx] += vx * dt_half;
        pos.y[idx] += vy * dt_half;
        pos.z[idx] += vz * dt_half;

        random_states[idx] = cs;
    }

    __global__ void baoab_step_two(
        md::DeviceVec3 vel,
        md::DeviceVec3 force,
        const int atom_count,
        const float dt_half_conv
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= atom_count) {
            return;
        }

        // 速度の更新3
        vel.x[idx] += force.x[idx] * dt_half_conv;
        vel.y[idx] += force.y[idx] * dt_half_conv;
        vel.z[idx] += force.z[idx] * dt_half_conv; 
    }
}

namespace md::integrators {
    LangevinIntegratorLJ::LangevinIntegratorLJ(
        float gamma, unsigned long long seed, TemperatureScheduler* scheduler)
        : gamma_(gamma), seed_(seed), scheduler_(scheduler) {
        if (!std::isfinite(gamma_) || gamma_ < 0.0f) {
            throw std::invalid_argument("Langevin gamma must be finite and non-negative");
        }
        if (scheduler_ == nullptr) {
            throw std::invalid_argument("Langevin integrator requires a temperature scheduler");
        }
    }

    void LangevinIntegratorLJ::init(const State& state, SimState& simstate) {
        init(state, simstate, seed_);
    }

    void LangevinIntegratorLJ::init(
        const State& state, 
        SimState& simstate, 
        unsigned long long seed
    ) {
        atom_count_ = state.n_atoms;
        dof_ = 3 * state.n_atoms;
        c1_ = std::exp(-this->gamma_ * simstate.dt);

        curand_state_.resize(state.n_atoms);

        thrust::for_each(
            thrust::device, 
            thrust::make_counting_iterator<int>(0), 
            thrust::make_counting_iterator<int>(state.n_atoms), 
            InitCurand(thrust::raw_pointer_cast(curand_state_.data()), seed)
        );
    }

    void LangevinIntegratorLJ::integrateStepOne(State& state, SimState& simstate) {
        // c3の計算
        scheduler_->get_temperature(state, simstate);

        const auto dt_half = simstate.dt * 0.5;
        const auto dt_half_conv = dt_half * conversion_factor;

        // 更新
        const int num_blocks = (atom_count_ + NUM_THREADS - 1) / NUM_THREADS;
        baoab_step_one<<<num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            state.pos, 
            state.vel, 
            state.force, 
            thrust::raw_pointer_cast(this->curand_state_.data()), 
            atom_count_, 
            dt_half, 
            dt_half_conv, 
            c1_, 
            boltzmann_constant
        );
    }

    void LangevinIntegratorLJ::integrateStepTwo(State& state, SimState& simstate) {
        const float half_dt_conv = 0.5f * simstate.dt * conversion_factor;

        const int blocks = (atom_count_ + NUM_THREADS - 1) / NUM_THREADS;
        baoab_step_two<<<blocks, NUM_THREADS, 0, simstate.stream>>>(
            state.vel, 
            state.force, 
            atom_count_, 
            half_dt_conv
        );
    }
}
