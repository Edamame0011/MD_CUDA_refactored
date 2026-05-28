#include <md/energy_minimizers/FireMinimizer.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <md/cells/CubicCell.cuh>
#include <md/core/constant.h>
#include <thrust/execution_policy.h>
#include <md/observers/Observer.cuh>

namespace {
    struct CalcDot {
        const dfloat3 vel, force;

        CalcDot(dfloat3 _v, dfloat3 _f) : vel(_v), force(_f) {}

        __device__ float operator() (const int idx) const {
            return vel.x[idx] * force.x[idx] + vel.y[idx] * force.y[idx] + vel.z[idx] * force.z[idx];
        }
    };

    struct Correct {
        dfloat3 pos, vel;
        float dt_half;
        Correct(dfloat3 p, dfloat3 v, float _dt_half) : pos(p), vel(v), dt_half(_dt_half) {}
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
        dfloat3 pos, vel;
        const dfloat3 force;
        const float* mass_inv;
        
        float dt;
        float dt_half_conv;
        float alpha;
        IntegrateStepOne(
            dfloat3 p, 
            dfloat3 v, 
            dfloat3 f, 
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
        const float dt_half_conv;

        dfloat3 vel;
        const dfloat3 force;
        const float* mass_inv;

        IntegrateStepTwo(
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

using namespace md::energy_minimizers;

template <typename CellType>
FireMinimizer<CellType>::FireMinimizer(
    State& _state, 
    CellType& _cell, 
    Interaction* _interaction, 
    Observer* _observer, 
    ConvChecker* _checker
) : state(_state), 
    cell(_cell), 
    interaction(_interaction), 
    observer(_observer), 
    checker(_checker) {}

template <typename CellType>
void FireMinimizer<CellType>::set_hyper_parameters(
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

template <typename CellType>
void FireMinimizer<CellType>::run() {
    auto N = state.n_atoms;
    auto view = state.get_view();

    interaction->calc_force(state);
    cudaMemset(view.vel.x, 0.0f, N * sizeof(float));
    cudaMemset(view.vel.y, 0.0f, N * sizeof(float));
    cudaMemset(view.vel.z, 0.0f, N * sizeof(float));

    int n_pos = 0;
    int n_neg = 0;
    float dt = dt_start;
    float alpha = alpha_start;
    float t = 0.0f;

    float dt_half_conv;

    // メインループ
    for (int i = 0; i < n_max; i ++) {
        float p = thrust::transform_reduce(
            thrust::device, 
            thrust::make_counting_iterator(0), 
            thrust::make_counting_iterator(N), 
            CalcDot(
                view.vel, 
                view.force
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
                thrust::device, 
                thrust::make_counting_iterator(0),  
                thrust::make_counting_iterator(N), 
                Correct(
                    view.pos, 
                    view.vel, 
                    dt * 0.5f
                )
            );
        }

        dt_half_conv = dt * 0.5f * conversion_factor;

        thrust::for_each(
            thrust::device, 
            thrust::make_counting_iterator(0), 
            thrust::make_counting_iterator(N), 
            IntegrateStepOne(
                view.pos, 
                view.vel, 
                view.force, 
                view.mass_inv, 
                dt, 
                dt_half_conv, 
                alpha
            )
        );
        cell.apply_pbc(state);
        interaction->calc_force(state);
        thrust::for_each(
            thrust::device, 
            thrust::make_counting_iterator(0), 
            thrust::make_counting_iterator(state.n_atoms), 
            IntegrateStepTwo(
                view.vel, 
                view.force, 
                view.mass_inv, 
                dt_half_conv
            )
        );

        t += dt;
        state.dt = dt;
        state.current_steps = i;

        observer->output(state);

        if (checker->check(state)) {
            md::observers::print_energies(state, interaction);
            break;
        }
    }
}

template class FireMinimizer<md::cells::CubicCell>;