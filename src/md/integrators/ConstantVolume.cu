#include <md/integrators/ConstantVolume.cuh>
#include <md/core/constant.h>

namespace {
    struct VelocityVerletStepOne {
        float dt;
        float conversion_factor;

        VelocityVerletStepOne(float _dt, float _conv) : dt(_dt), conversion_factor(_conv) {}

        template <typename Tuple>
        __host__ __device__ void operator() (Tuple t) {
            auto& pos_x = thrust::get<0>(t);
            auto& pos_y = thrust::get<1>(t);
            auto& pos_z = thrust::get<2>(t);

            auto& vel_x = thrust::get<3>(t);
            auto& vel_y = thrust::get<4>(t);
            auto& vel_z = thrust::get<5>(t);

            const auto force_x = thrust::get<6>(t);
            const auto force_y = thrust::get<7>(t);
            const auto force_z = thrust::get<8>(t);

            const auto mass = thrust::get<9>(t);

            float mass_inv = 1.0f / mass;
            float dt_half_conv = 0.5f * dt * conversion_factor;

            // 速度の更新
            vel_x += force_x * mass_inv * dt_half_conv;
            vel_y += force_y * mass_inv * dt_half_conv;
            vel_z += force_z * mass_inv * dt_half_conv;

            // 位置の更新
            pos_x += vel_x * dt;
            pos_y += vel_y * dt;
            pos_z += vel_z * dt;
        }
    };

    struct VelocityVerletStepTwo {
        float dt;
        float conversion_factor;

        VelocityVerletStepTwo(float _dt, float _conv) : dt(_dt), conversion_factor(_conv) {}

        template <typename Tuple>
        __host__ __device__ void operator() (Tuple t) {
            auto& vel_x = thrust::get<0>(t);
            auto& vel_y = thrust::get<1>(t);
            auto& vel_z = thrust::get<2>(t);

            const auto force_x = thrust::get<3>(t);
            const auto force_y = thrust::get<4>(t);
            const auto force_z = thrust::get<5>(t);

            const auto mass = thrust::get<6>(t);

            float mass_inv = 1.0f / mass;
            float dt_half_conv = 0.5f * dt * conversion_factor;

            // 速度の更新
            vel_x += force_x * mass_inv * dt_half_conv;
            vel_y += force_y * mass_inv * dt_half_conv;
            vel_z += force_z * mass_inv * dt_half_conv; 
        }
    };
}

using namespace md::integrators;

void ConstantVolume::integrateStepOne(State& state) {
    // 熱浴の更新
    this->thermostat->stepOne(state);

    // zip
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            state.d_positions.x.begin(), 
            state.d_positions.y.begin(), 
            state.d_positions.z.begin(), 
            state.d_velocities.x.begin(), 
            state.d_velocities.y.begin(), 
            state.d_velocities.z.begin(), 
            state.d_forces.x.begin(), 
            state.d_forces.y.begin(), 
            state.d_forces.z.begin(), 
            state.d_masses.begin()
        )
    );

    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(
            state.d_positions.x.end(), 
            state.d_positions.y.end(), 
            state.d_positions.z.end(), 
            state.d_velocities.x.end(), 
            state.d_velocities.y.end(), 
            state.d_velocities.z.end(), 
            state.d_forces.x.end(), 
            state.d_forces.y.end(), 
            state.d_forces.z.end(), 
            state.d_masses.end()
        )
    );

    // 更新
    thrust::for_each(
        zip_begin, 
        zip_end, 
        VelocityVerletStepOne(state.dt, conversion_factor)
    );


}

void ConstantVolume::integrateStepTwo(State& state) {
    // zip
    auto zip_vel_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            state.d_velocities.x.begin(), 
            state.d_velocities.y.begin(), 
            state.d_velocities.z.begin(), 
            state.d_forces.x.begin(), 
            state.d_forces.y.begin(), 
            state.d_forces.z.begin(), 
            state.d_masses.begin()
        )
    );

    auto zip_vel_end = thrust::make_zip_iterator(
        thrust::make_tuple(
            state.d_velocities.x.end(), 
            state.d_velocities.y.end(), 
            state.d_velocities.z.end(), 
            state.d_forces.x.end(), 
            state.d_forces.y.end(), 
            state.d_forces.z.end(), 
            state.d_masses.end()
        )
    );

    // 更新
    thrust::for_each(zip_vel_begin, zip_vel_end, VelocityVerletStepTwo(state.dt, conversion_factor));

    // 熱浴の更新
    this->thermostat->stepTwo(state);
}