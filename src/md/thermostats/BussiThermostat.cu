#include <md/thermostats/BussiThermostat.cuh>
#include <md/core/constant.h>
#include <thrust/transform_reduce.h>
#include <md/utils/compute.cuh>

namespace {
    struct Square {
        __host__ __device__ float operator() (float rand) {
            return rand * rand;
        }
    };

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
    if (state.n_atoms % 2 != 0) throw std::runtime_error("Bussi熱浴を利用する場合、原子数は偶数である必要があります。");
    this->dof = 3 * state.n_atoms;

    this->random.resize(dof);
}

void BussiThermostat::generate_rand(const State& state) {
    curandGenerateNormal(this->gen, thrust::raw_pointer_cast(this->random.data()), this->dof, 0.0f, 1.0f);}

void BussiThermostat::stepTwo(State& state) {
    float target_temperature = this->scheduler->get_temperature(state.current_steps);
    float targ_kin = (dof * boltzmann_constant * target_temperature) / 2;
    float kinetic_energy = md::utils::compute::calc_kinetic_energy(state);

    this->generate_rand(state);

    // スケーリング要素を計算
    float sum_rand2 = thrust::transform_reduce(
        this->random.begin(), 
        this->random.end(), 
        Square(), 
        0.0f, 
        thrust::plus<float>()
    );
    float f = std::exp(-state.dt / this->tau);
    float r = this->random[0];
    float alpha2 = f + (targ_kin * (1 - f) * sum_rand2) / (dof * kinetic_energy) + 2 * r * std::sqrt((targ_kin * f * (1 - f)) / (dof * kinetic_energy));

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