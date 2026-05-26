#include <md/thermostats/KinEnergyCalculator.cuh>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {
    struct CalcKinEnergy {
        const float *vel_x, *vel_y, *vel_z;
        const float *mass;

        CalcKinEnergy(const float *vx, const float *vy, const float *vz, const float *m)
            : vel_x(vx), vel_y(vy), vel_z(vz), mass(m) {}

        __device__ float operator() (int idx) const {
            const auto vx = vel_x[idx];
            const auto vy = vel_y[idx];
            const auto vz = vel_z[idx];
            const auto m = mass[idx];

            return 0.5f * m * (vx * vx + vy * vy + vz * vz);
        }
    };
}

using namespace md::thermostats;

KinEnergyCalculator::KinEnergyCalculator(State& state) {
    auto view = state.get_view();
    auto N = state.n_atoms;

    CalcKinEnergy op(view.vel.x, view.vel.y, view.vel.z, view.mass);
    thrust::counting_iterator<int> count_itr(0);

    auto trans_itr = thrust::make_transform_iterator(count_itr, op);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, trans_itr, view.kinetic_energy, N);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

KinEnergyCalculator::~KinEnergyCalculator() {
    cudaFree(d_temp_storage);
}

void KinEnergyCalculator::calc_kinetic_energy(State& state) {
    auto view = state.get_view();
    auto N = state.n_atoms;

    CalcKinEnergy op(view.vel.x, view.vel.y, view.vel.z, view.mass);
    thrust::counting_iterator<int> count_itr(0);

    auto trans_itr = thrust::make_transform_iterator(count_itr, op);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, trans_itr, view.kinetic_energy, N, state.stream);
}