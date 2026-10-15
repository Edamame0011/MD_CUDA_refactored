#include <md/energy_minimizers/FireMinimizer.hpp>

#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#include <md/core/constant.h>
#include <md/core/State.hpp>
#include <md/interactions/Interaction.hpp>
#include <md/integrators/Integrator.hpp>
#include <md/observers/Observer.hpp>
#include <md/convergence_checkers/ConvChecker.hpp>
#include <md/core/Cell.cuh>

using DeviceVec3 = md::DeviceVec3;

namespace {
    struct CalcDot {
        const DeviceVec3 vel, force;

        CalcDot(DeviceVec3 v, DeviceVec3 f) : vel(v), force(f) {}

        __device__ float operator() (const int idx) const {
            return vel.x[idx] * force.x[idx] + vel.y[idx] * force.y[idx] + vel.z[idx] * force.z[idx];
        }
    };

    struct Correct {
        DeviceVec3 pos, vel;
        float dt_half;
        Correct(DeviceVec3 p, DeviceVec3 v, float _dt_half) : pos(p), vel(v), dt_half(_dt_half) {}
        __device__ void operator() (const int idx) {
            const auto px = pos.x[idx] - dt_half * vel.x[idx];
            const auto py = pos.y[idx] - dt_half * vel.y[idx];
            const auto pz = pos.z[idx] - dt_half * vel.z[idx];

            pos.x[idx] = px;
            pos.y[idx] = py;
            pos.z[idx] = pz;
            
            vel.x[idx] = 0.0f;
            vel.y[idx] = 0.0f;
            vel.z[idx] = 0.0f;
        }
    };

    struct IntegrateStepOne {
        DeviceVec3 pos, vel;
        const DeviceVec3 force;
        const float* mass_inv;
        
        float dt;
        float dt_half_conv;
        float alpha;
        IntegrateStepOne(
            DeviceVec3 p, 
            DeviceVec3 v, 
            DeviceVec3 f, 
            const float* mi, 
            float _dt, 
            float _dt_half_conv, 
            float _alpha
        ) : pos(p), vel(v), force(f), mass_inv(mi), dt(_dt), dt_half_conv(_dt_half_conv), alpha(_alpha) {}

        __device__ void operator() (const int idx) {
            const auto mi = mass_inv[idx];

            const auto fx = force.x[idx];
            const auto fy = force.y[idx];
            const auto fz = force.z[idx];

            auto vx = vel.x[idx] + fx * mi * dt_half_conv;
            auto vy = vel.y[idx] + fy * mi * dt_half_conv;
            auto vz = vel.z[idx] + fz * mi * dt_half_conv;

            float v_norm = sqrtf(vx * vx + vy * vy + vz * vz);
            float f_norm = sqrtf(fx * fx + fy * fy + fz * fz);

            auto f_norm_inv = (f_norm > 1e-12f) ? (1.0f / f_norm) : 0.0f;

            vx = (1 - alpha) * vx + alpha * fx * v_norm * f_norm_inv;
            vy = (1 - alpha) * vy + alpha * fy * v_norm * f_norm_inv;
            vz = (1 - alpha) * vz + alpha * fz * v_norm * f_norm_inv;

            pos.x[idx] += vx * dt;
            pos.y[idx] += vy * dt;
            pos.z[idx] += vz * dt;

            vel.x[idx] = vx;
            vel.y[idx] = vy;
            vel.z[idx] = vz;
        }
    };

    struct IntegrateStepTwo {
        const float dt_half_conv_;

        DeviceVec3 vel_;
        const DeviceVec3 force_;
        const float* mass_inv_;

        IntegrateStepTwo(
            DeviceVec3 v, 
            DeviceVec3 f, 
            const float* mass_inv, 
            float dt_half_conv
        ) : vel_(v), force_(f), mass_inv_(mass_inv), dt_half_conv_(dt_half_conv) {}

        __host__ __device__ void operator() (int idx) {
            const auto mi = mass_inv_[idx];

            // 速度の更新
            vel_.x[idx] += force_.x[idx] * mi * dt_half_conv_;
            vel_.y[idx] += force_.y[idx] * mi * dt_half_conv_;
            vel_.z[idx] += force_.z[idx] * mi * dt_half_conv_;
        }
    };
}

using namespace md::energy_minimizers;

FireMinimizer::FireMinimizer(
    State* state, 
    SimState& simstate, 
    Cell* cell, 
    Interaction* interaction, 
    Observer* observer, 
    ConvChecker* checker
) : state_(state), 
    simstate_(simstate), 
    cell_(cell), 
    interaction_(interaction), 
    observer_(observer), 
    checker_(checker) {}

void FireMinimizer::set_hyper_parameters(
    int _n_max, 
    int _n_delay, 
    int _n_neg_max, 
    float _dt_start, 
    float _t_max, 
    float _t_min, 
    float _f_inc, 
    float _f_dec, 
    float _alpha_start, 
    float _f_alpha, 
    bool _initialdelay
) {
    n_max = _n_max;
    n_delay = _n_delay;
    n_neg_max = _n_neg_max;
    dt_start = _dt_start;
    t_max = _t_max;
    t_min = _t_min;
    f_inc = _f_inc;
    f_dec = _f_dec;
    alpha_start = _alpha_start;
    f_alpha = _f_alpha;
    initialdelay = _initialdelay;
}

void FireMinimizer::run() {
    auto N = state_->n_atoms;

    interaction_->calc_force(state_, simstate_);
    cudaMemsetAsync(state_->vel.x, 0.0f, N * sizeof(float), simstate_.stream);
    cudaMemsetAsync(state_->vel.y, 0.0f, N * sizeof(float), simstate_.stream);
    cudaMemsetAsync(state_->vel.z, 0.0f, N * sizeof(float), simstate_.stream);

    int n_pos = 0;
    int n_neg = 0;
    float dt = dt_start;
    float alpha = alpha_start;
    float t = 0.0f;

    float dt_half_conv;

    // メインループ
    for (int i = 0; i < n_max; i ++) {
        float p = thrust::transform_reduce(
            thrust::cuda::par.on(simstate_.stream), 
            thrust::make_counting_iterator<size_t>(0), 
            thrust::make_counting_iterator<size_t>(N), 
            CalcDot(
                state_->vel, 
                state_->force
            ), 
            0.0f, 
            thrust::plus()
        );
        if (p > 0) {
            n_pos ++;
            n_neg = 0;
            if (n_pos > n_delay) {
                dt = std::min(dt * f_inc, dt_start * t_max);
                alpha = alpha * f_alpha;
            }
        } else {
            n_pos = 0;
            n_neg ++;
            if (n_neg > n_neg_max) break;
            if (!(initialdelay && (i < n_delay))) {
                dt = std::max(dt * f_dec, dt_start * t_min);
                alpha = alpha_start;
            }
            thrust::for_each(
                thrust::cuda::par.on(simstate_.stream), 
                thrust::make_counting_iterator<size_t>(0),  
                thrust::make_counting_iterator<size_t>(N), 
                Correct(
                    state_->pos, 
                    state_->vel, 
                    dt * 0.5f
                )
            );
        }

        dt_half_conv = dt * 0.5f * conversion_factor;

        thrust::for_each(
            thrust::cuda::par.on(simstate_.stream), 
            thrust::make_counting_iterator<int>(0), 
            thrust::make_counting_iterator<int>(N), 
            IntegrateStepOne(
                state_->pos, 
                state_->vel, 
                state_->force, 
                state_->mass_inv, 
                dt, 
                dt_half_conv, 
                alpha
            )
        );
        cell_->apply_pbc(state_, simstate_);
        interaction_->calc_force(state_, simstate_);
        thrust::for_each(
            thrust::cuda::par.on(simstate_.stream), 
            thrust::make_counting_iterator<int>(0), 
            thrust::make_counting_iterator<int>(state_->n_atoms), 
            IntegrateStepTwo(
                state_->vel, 
                state_->force, 
                state_->mass_inv, 
                dt_half_conv
            )
        );

        t += dt;
        simstate_.dt = dt;
        simstate_.current_steps = i;

        observer_->output(state_, simstate_);

        if (checker_->check(state_, simstate_)) {
            break;
        }
    }
}