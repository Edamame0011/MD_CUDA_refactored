#include <md/integrators/LangevinIntegrator.cuh>
#include <md/core/constant.h>

namespace {
    struct StepOne {
        float dt;
        float conversion_factor;
        float c1, c3;

        float *pos_x, *pos_y, *pos_z;
        float *vel_x, *vel_y, *vel_z;
        const float *force_x, *force_y, *force_z;
        const float *mass;
        const float *rand_x, *rand_y, *rand_z;

        StepOne(float _dt, float _conversion_factor, float _c1, float _c3, 
                float* _px, float* _py, float* _pz,
                float* _vx, float* _vy, float* _vz,
                const float* _fx, const float* _fy, const float* _fz,
                const float* _m,
                const float* _rx, const float* _ry, const float* _rz)
         : dt(_dt), conversion_factor(_conversion_factor), c1(_c1), c3(_c3), 
           pos_x(_px), pos_y(_py), pos_z(_pz),
           vel_x(_vx), vel_y(_vy), vel_z(_vz),
           force_x(_fx), force_y(_fy), force_z(_fz), mass(_m),
           rand_x(_rx), rand_y(_ry), rand_z(_rz) {}

        __host__ __device__ void operator() (const int i) {        
            float mass_inv = 1.0f / mass[i];
            float mass_sq_inv = rsqrtf(mass[i]);
            float dt_half = 0.5f * dt; 
            float dt_half_conv = 0.5f * dt * conversion_factor;

            // 速度の更新1
            vel_x[i] += force_x[i] * mass_inv * dt_half_conv;
            vel_y[i] += force_y[i] * mass_inv * dt_half_conv;
            vel_z[i] += force_z[i] * mass_inv * dt_half_conv;

            // 位置の更新1
            pos_x[i] += vel_x[i] * dt_half;
            pos_y[i] += vel_y[i] * dt_half;
            pos_z[i] += vel_z[i] * dt_half;

            // 速度の更新2
            vel_x[i] = c1 * vel_x[i] + c3 * mass_sq_inv * rand_x[i];
            vel_y[i] = c1 * vel_y[i] + c3 * mass_sq_inv * rand_y[i];
            vel_z[i] = c1 * vel_z[i] + c3 * mass_sq_inv * rand_z[i];

            // 位置の更新2
            pos_x[i] += vel_x[i] * dt_half;
            pos_y[i] += vel_y[i] * dt_half;
            pos_z[i] += vel_z[i] * dt_half;
        }
    };   

    struct StepTwo {
        float dt;
        float conversion_factor;

        StepTwo(float _dt, float _conv) : dt(_dt), conversion_factor(_conv) {}

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

            // 速度の更新3
            vel_x += force_x * mass_inv * dt_half_conv;
            vel_y += force_y * mass_inv * dt_half_conv;
            vel_z += force_z * mass_inv * dt_half_conv; 
        }
    };
}

using namespace md::integrators;

void LangevinIntegrator::init(const State& state) {
    if (state.n_atoms % 2 != 0) throw std::runtime_error("Langevin熱浴を利用する場合、原子数は偶数である必要があります。");
    this->dof = 3 * state.n_atoms;
    this->c1 = std::exp(-this->gamma * state.dt);

    this->random.x.resize(state.n_atoms);
    this->random.y.resize(state.n_atoms);
    this->random.z.resize(state.n_atoms);
}

void LangevinIntegrator::init_random(const State& state) {
    curandGenerateNormal(this->gen, thrust::raw_pointer_cast(this->random.x.data()), state.n_atoms, 0.0f, 1.0f);
    curandGenerateNormal(this->gen, thrust::raw_pointer_cast(this->random.y.data()), state.n_atoms, 0.0f, 1.0f);
    curandGenerateNormal(this->gen, thrust::raw_pointer_cast(this->random.z.data()), state.n_atoms, 0.0f, 1.0f);
}

void LangevinIntegrator::integrateStepOne(State& state) {
    // c3の計算
    float target_temperature = this->scheduler->get_temperature(state.current_steps);
    float c3 = std::sqrt(boltzmann_constant * target_temperature * (1 - c1 * c1));

    // 乱数の更新
    this->init_random(state);

    // 更新
    thrust::for_each(
        thrust::make_counting_iterator<int>(0), 
        thrust::make_counting_iterator<int>(state.n_atoms), 
        StepOne(
            state.dt, 
            conversion_factor, 
            this->c1, 
            c3, 
            thrust::raw_pointer_cast(state.d_positions.x.data()), 
            thrust::raw_pointer_cast(state.d_positions.y.data()), 
            thrust::raw_pointer_cast(state.d_positions.z.data()), 
            thrust::raw_pointer_cast(state.d_velocities.x.data()), 
            thrust::raw_pointer_cast(state.d_velocities.y.data()), 
            thrust::raw_pointer_cast(state.d_velocities.z.data()), 
            thrust::raw_pointer_cast(state.d_forces.x.data()), 
            thrust::raw_pointer_cast(state.d_forces.y.data()), 
            thrust::raw_pointer_cast(state.d_forces.z.data()), 
            thrust::raw_pointer_cast(state.d_masses.data()), 
            thrust::raw_pointer_cast(this->random.x.data()), 
            thrust::raw_pointer_cast(this->random.y.data()), 
            thrust::raw_pointer_cast(this->random.z.data())
        )
    );
}

void LangevinIntegrator::integrateStepTwo(State& state) {
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(
            state.d_velocities.x.begin(), 
            state.d_velocities.y.begin(), 
            state.d_velocities.z.begin(), 
            state.d_forces.x.begin(), 
            state.d_forces.y.begin(), 
            state.d_forces.z.begin(), 
            state.d_masses.begin()
        )), 
        thrust::make_zip_iterator(thrust::make_tuple(
            state.d_velocities.x.end(), 
            state.d_velocities.y.end(), 
            state.d_velocities.z.end(), 
            state.d_forces.x.end(), 
            state.d_forces.y.end(), 
            state.d_forces.z.end(), 
            state.d_masses.end()
        )), 
        StepTwo(state.dt, conversion_factor)
    );
}