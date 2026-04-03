#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
    struct Multiply {
        float* __restrict__ v;
        float* __restrict__ m;

        Multiply(float* _v, float* _m) : v(_v), m(_m) {}

        __host__ __device__ float operator() (int idx) const {
            return v[idx] * m[idx];
        }
    };

    struct RemoveDrift {
        dfloat3 vel;
        float avg_x, avg_y, avg_z;
        RemoveDrift(
            dfloat3 _vel, 
            float _avg_x, 
            float _avg_y, 
            float _avg_z
        ) : vel(_vel), 
            avg_x(_avg_x), 
            avg_y(_avg_y), 
            avg_z(_avg_z) {}

        __host__ __device__ void operator() (int idx) const {
            vel.x[idx] -= avg_x;
            vel.y[idx] -= avg_y;
            vel.z[idx] -= avg_z;
        }
    };

    struct CalcKinEnergy {
        dfloat3 vel;
        float* __restrict__ mass;

        CalcKinEnergy(
            dfloat3 _vel, 
            float* _mass
        ) : vel(_vel), mass(_mass) {}

        __host__ __device__ float operator() (int idx) {
            auto vx = vel.x[idx];
            auto vy = vel.y[idx];
            auto vz = vel.z[idx];

            return 0.5 * mass[idx] * (vx * vx + vy * vy + vz * vz);
        }
    };
}

void md::utils::compute::remove_drift(State& state) {
    md::StateView view = state.get_view();
    int N = state.n_atoms;

    // calc drift
    float weighted_sum_x = thrust::transform_reduce(
        thrust::device, 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        Multiply(
            view.vel.x, 
            view.mass
        ), 
        0.0f, 
        thrust::plus<float>());
    float weighted_sum_y = thrust::transform_reduce(
        thrust::device, 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        Multiply(
            view.vel.y, 
            view.mass
        ), 
        0.0f, 
        thrust::plus<float>());
    float weighted_sum_z = thrust::transform_reduce(
        thrust::device, 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        Multiply(
            view.vel.z, 
            view.mass
        ), 
        0.0f, 
        thrust::plus<float>());

    float mass_sum = thrust::reduce(
        thrust::device, 
        view.mass, 
        view.mass + N, 
        0.0f, 
        thrust::plus<float>()
    );

    float avg_x = weighted_sum_x / mass_sum;
    float avg_y = weighted_sum_y / mass_sum;
    float avg_z = weighted_sum_z / mass_sum;

    // remove drift
    thrust::for_each(
        thrust::device, 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        RemoveDrift(
            view.vel, 
            avg_x, 
            avg_y, 
            avg_z
        )
    );
}

float md::utils::compute::calc_kinetic_energy(State& state) {
    auto view = state.get_view();
    auto N = state.n_atoms;

    // 運動エネルギーの計算
    auto kinetic_energy = thrust::transform_reduce(
        thrust::device, 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N),  
        CalcKinEnergy(
            view.vel, 
            view.mass
        ), 
        0.0f, 
        thrust::plus<float>()
    );

    return kinetic_energy / conversion_factor;
}