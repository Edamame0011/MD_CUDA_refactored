#include <md/thermostats/BussiThermostat.cuh>
#include <md/core/constant.h>
#include <thrust/transform_reduce.h>
#include <md/utils/compute.cuh>

namespace {
    struct Scaling {
        float scaling_factor;

        Scaling(float _scaling_factor) : scaling_factor(_scaling_factor) {}

        template <typename Tuple>
        __host__ __device__ void operator() (Tuple t) const {
            thrust::get<0>(t) *= scaling_factor;
            thrust::get<1>(t) *= scaling_factor;
            thrust::get<2>(t) *= scaling_factor;
        }
    };
}

using namespace md::thermostats;

void BussiThermostat::init(const State& state) {
    this->dof = 3 * state.n_atoms;
    this->gamma_dist = std::gamma_distribution<float>((dof - 2) / 2.0f, 1.0f);
}

void BussiThermostat::stepTwo(State& state) {
    float target_temperature = this->scheduler->get_temperature(state.current_steps);
    float targ_kin = (dof * boltzmann_constant * target_temperature) / 2;
    float kinetic_energy = md::utils::compute::calc_kinetic_energy(state);

    // 乱数を生成
    float r1 = this->normal_dist(this->gen);
    float r2 = this->normal_dist(this->gen);
    float g = this->gamma_dist(this->gen);

    // スケーリング要素を計算
    float f = std::exp(-state.dt / this->tau);
    float alpha2 = f + (targ_kin * (1 - f) * (r1 * r1 + r2 * r2 + 2 * g)) / (dof * kinetic_energy) + 2 * r1 * std::sqrt((targ_kin * f * (1 - f)) / (dof * kinetic_energy));

    // スケーリング
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(
            state.d_velocities.x.begin(), 
            state.d_velocities.y.begin(), 
            state.d_velocities.z.begin()
        )), 
        thrust::make_zip_iterator(thrust::make_tuple(
            state.d_velocities.x.end(), 
            state.d_velocities.y.end(), 
            state.d_velocities.z.end()
        )), 
        Scaling(std::sqrt(alpha2))
    );
}