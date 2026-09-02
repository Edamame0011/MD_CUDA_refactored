#include <md/integrators/ConstantVolumeLJ.hpp>

#include <md/core/constant.h>
#include <md/core/State.hpp>
#include <md/thermostats/Thermostat.cuh>

using DeviceVec3 = md::DeviceVec3;

namespace {
    __global__ void velocity_verlet_stepone(
        const float dt, 
        const float dt_half_conv, 
        const int n_atoms, 
        DeviceVec3 pos, 
        DeviceVec3 vel, 
        const DeviceVec3 force
    ) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_atoms) return;

        // 速度の更新
        const auto vx = vel.x[idx] + force.x[idx] * dt_half_conv;
        const auto vy = vel.y[idx] + force.y[idx] * dt_half_conv;
        const auto vz = vel.z[idx] + force.z[idx] * dt_half_conv;

        vel.x[idx] = vx;
        vel.y[idx] = vy;
        vel.z[idx] = vz;

        // 位置の更新
        pos.x[idx] += vx * dt;
        pos.y[idx] += vy * dt;
        pos.z[idx] += vz * dt;
    }

    __global__ void velocity_verlet_steptwo(
        const float dt_half_conv, 
        const int n_atoms, 
        DeviceVec3 vel, 
        const DeviceVec3 force
    ) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_atoms) return;

        // 速度の更新
        vel.x[idx] += force.x[idx] * dt_half_conv;
        vel.y[idx] += force.y[idx] * dt_half_conv;
        vel.z[idx] += force.z[idx] * dt_half_conv;
    }
}

namespace md::integrators {    
    void ConstantVolumeLJ::integrateStepOne(State& state, SimState& simstate) {
        // 熱浴の更新
        this->thermostat->stepOne(state, simstate);

        const auto N = state.n_atoms;
        const auto dt = simstate.dt;
        const auto dt_half_conv = dt * 0.5 * conversion_factor;

        // 更新
        int num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;
        velocity_verlet_stepone<<<num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            dt, 
            dt_half_conv, 
            N, 
            state.pos, 
            state.vel, 
            state.force
        );
    }

    void ConstantVolumeLJ::integrateStepTwo(State& state, SimState& simstate) {
        const auto dt_half_conv = simstate.dt * 0.5 * conversion_factor;
        const auto N = state.n_atoms;

        // 更新
        int num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;
        velocity_verlet_steptwo<<<num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            dt_half_conv, 
            N, 
            state.vel, 
            state.force
        );

        // 熱浴の更新
        this->thermostat->stepTwo(state, simstate);
    }
}
