#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <thrust/transform_reduce.h>

namespace {
    struct Multiply {
        template <typename Tuple>
        __host__ __device__ float operator() (Tuple t) const {
            return thrust::get<0>(t) * thrust::get<1>(t);
        }
    };

    struct RemoveDrift {
        float avg_x, avg_y, avg_z;
        RemoveDrift(float _avg_x, float _avg_y, float _avg_z) : avg_x(_avg_x), avg_y(_avg_y), avg_z(_avg_z) {}
        template <typename Tuple>
        __host__ __device__ auto operator() (Tuple vel) const {
            return thrust::make_tuple(
                thrust::get<0>(vel) - avg_x, 
                thrust::get<1>(vel) - avg_y, 
                thrust::get<2>(vel) - avg_z
            );
        }
    };

    struct CalcKinEnergy {
        template <typename Tuple>
        __host__ __device__ float operator() (Tuple t) {
            float vel_x = thrust::get<0>(t);
            float vel_y = thrust::get<1>(t);
            float vel_z = thrust::get<2>(t);
            
            return 0.5 * thrust::get<3>(t) * (vel_x * vel_x + vel_y * vel_y + vel_z * vel_z);
        }
    };
}


void md::utils::compute::remove_drift(State& state) {
    // zip
    auto zip_begin_x = thrust::make_zip_iterator(thrust::make_tuple(state.d_velocities.x.begin(), state.d_masses.begin()));
    auto zip_end_x = thrust::make_zip_iterator(thrust::make_tuple(state.d_velocities.x.end(), state.d_masses.end()));

    auto zip_begin_y = thrust::make_zip_iterator(thrust::make_tuple(state.d_velocities.y.begin(), state.d_masses.begin()));
    auto zip_end_y = thrust::make_zip_iterator(thrust::make_tuple(state.d_velocities.y.end(), state.d_masses.end()));

    auto zip_begin_z = thrust::make_zip_iterator(thrust::make_tuple(state.d_velocities.z.begin(), state.d_masses.begin()));
    auto zip_end_z = thrust::make_zip_iterator(thrust::make_tuple(state.d_velocities.z.end(), state.d_masses.end()));

    // calc drift
    float weighted_sum_x = thrust::transform_reduce(zip_begin_x, zip_end_x, Multiply(), 0.0f, thrust::plus<float>());
    float weighted_sum_y = thrust::transform_reduce(zip_begin_y, zip_end_y, Multiply(), 0.0f, thrust::plus<float>());
    float weighted_sum_z = thrust::transform_reduce(zip_begin_z, zip_end_z, Multiply(), 0.0f, thrust::plus<float>());

    float mass_sum = thrust::reduce(state.d_masses.begin(), state.d_masses.end(), 0.0f, thrust::plus<float>());

    float avg_x = weighted_sum_x / mass_sum;
    float avg_y = weighted_sum_y / mass_sum;
    float avg_z = weighted_sum_z / mass_sum;

    // remove drift
    auto zip_vel_begin = thrust::make_zip_iterator(thrust::make_tuple(state.d_velocities.x.begin(), state.d_velocities.y.begin(), state.d_velocities.z.begin()));
    auto zip_vel_end = thrust::make_zip_iterator(thrust::make_tuple(state.d_velocities.x.end(), state.d_velocities.y.end(), state.d_velocities.z.end()));
    thrust::transform(zip_vel_begin, zip_vel_end, zip_vel_begin, RemoveDrift(avg_x, avg_y, avg_z));
}

float md::utils::compute::calc_kinetic_energy(const State& state) {
    // zip
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
        state.d_velocities.x.begin(), state.d_velocities.y.begin(), state.d_velocities.z.begin(), state.d_masses.begin()
    ));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(
        state.d_velocities.x.end(), state.d_velocities.y.end(), state.d_velocities.z.end(), state.d_masses.end()
    ));

    // 運動エネルギーの計算
    float kinetic_energy = thrust::transform_reduce(
        zip_begin, 
        zip_end, 
        CalcKinEnergy(), 
        0.0f, 
        thrust::plus<float>()
    );

    return kinetic_energy / conversion_factor;
}