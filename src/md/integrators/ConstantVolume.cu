#include <md/integrators/ConstantVolume.cuh>

#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/thermostats/Thermostat.cuh>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
    struct VelocityVerletStepOne {
        const float dt;
        const float dt_half_conv;

        dfloat3 pos, vel;
        const dfloat3 force;
        const float* mass_inv;

        VelocityVerletStepOne(
            dfloat3 _pos, 
            dfloat3 _vel, 
            dfloat3 _force, 
            float *_mass_inv, 
            float _dt, 
            float _dt_half_conv
        ) : pos(_pos), vel(_vel), force(_force), mass_inv(_mass_inv), dt(_dt), dt_half_conv(_dt_half_conv) {}

        __host__ __device__ void operator() (int idx) {
            const auto mi = mass_inv[idx];

            // 速度の更新
            const auto vx = vel.x[idx] + force.x[idx] * mi * dt_half_conv;
            const auto vy = vel.y[idx] + force.y[idx] * mi * dt_half_conv;
            const auto vz = vel.z[idx] + force.z[idx] * mi * dt_half_conv;

            vel.x[idx] = vx;
            vel.y[idx] = vy;
            vel.z[idx] = vz;

            // 位置の更新
            pos.x[idx] += vx * dt;
            pos.y[idx] += vy * dt;
            pos.z[idx] += vz * dt;
        }
    };

    struct VelocityVerletStepTwo {
        const float dt_half_conv;

        dfloat3 vel;
        const dfloat3 force;
        const float* mass_inv;

        VelocityVerletStepTwo(
            dfloat3 _vel, 
            dfloat3 _force, 
            const float* _mass_inv, 
            float _dt_half_conv
        ) : vel(_vel), force(_force), mass_inv(_mass_inv), dt_half_conv(_dt_half_conv) {}

        __host__ __device__ void operator() (int idx) {
            const auto mi = mass_inv[idx];

            // 速度の更新
            vel.x[idx] += force.x[idx] * mi * dt_half_conv;
            vel.y[idx] += force.y[idx] * mi * dt_half_conv;
            vel.z[idx] += force.z[idx] * mi * dt_half_conv;
        }
    };
}

using namespace md::integrators;

void ConstantVolume::integrateStepOne(State& state) {
    // 熱浴の更新
    this->thermostat->stepOne(state);

    const auto dt = state.dt;
    const auto dt_half_conv = dt * 0.5 * conversion_factor;

    // 更新
    thrust::for_each(
        thrust::cuda::par_nosync.on(state.stream), 
        thrust::make_counting_iterator<int>(0), 
        thrust::make_counting_iterator<int>(state.n_atoms), 
        VelocityVerletStepOne(
            state.pos, 
            state.vel, 
            state.force, 
            state.mass_inv,  
            dt, 
            dt_half_conv
        )
    );
}

void ConstantVolume::integrateStepTwo(State& state) {
    const auto dt_half_conv = state.dt * 0.5 * conversion_factor;

    // 更新
    thrust::for_each(
        thrust::cuda::par_nosync.on(state.stream), 
        thrust::make_counting_iterator<int>(0), 
        thrust::make_counting_iterator<int>(state.n_atoms), 
        VelocityVerletStepTwo(
            state.vel, 
            state.force, 
            state.mass_inv, 
            dt_half_conv
        )
    );

    // 熱浴の更新
    this->thermostat->stepTwo(state);
}