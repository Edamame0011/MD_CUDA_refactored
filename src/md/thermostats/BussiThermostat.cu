#include <md/thermostats/BussiThermostat.cuh>

#include <md/core/constant.h>
#include <md/thermostats/KinEnergyCalculator.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/core/State.cuh>
#include <md/utils/compute.cuh>

#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
    // Marsaglia and Tsang's Method (https://daannoordenbos.github.io/gamma-sampling/ を参考に書きました。)
    __device__ float generate_gamma(
        curandState* state, 
        float k, 
        float theta
    ) {
        float d = k - 1.0f / 3.0f;
        float c = 1.0f / sqrtf(9.0f * d);
        float x, v, u, x_sq;

        while (1) {
            v = -1.0f;
            while(v <= 0.0f) {
                x = curand_normal(state);
                v = 1.0f + c * x;
            }
            v = v * v * v;
            u = curand_uniform(state);
            x_sq = x * x;
            if (u < 1.0f - 0.0331f * x_sq * x_sq || logf(u) < 0.5f * x_sq + d * (1.0f - v + logf(v))) {
                return d * v * theta;
            }
        }
    }

    __global__ void calc_scaling_factor(
        float *kinetic_energy, 
        float *scaling_factor, 
        curandState* state, 
        float dof, 
        float tau_inv, 
        float boltzmann_constant, 
        float dt  
    ) {
        float targ_kin = (dof * boltzmann_constant * md::c_target_temperature) / 2.0f;
        float r1 = curand_normal(state);
        float r2 = curand_normal(state);
        float g = generate_gamma(state, (dof - 2.0f) / 2.0f, 1.0f);
        float current_kin = *kinetic_energy;
        float f = expf(-dt * tau_inv);
        float alpha2 = f + (targ_kin * (1.0f - f) * (r1 * r1 + r2 * r2 + 2 * g)) / (dof * current_kin) + 2.0f * r1 * sqrtf((targ_kin * f * (1.0f - f)) / (dof * current_kin));

        *scaling_factor = sqrtf(alpha2);
    }

    __global__ void init_curand(
        curandState* state, 
        unsigned long long seed
    ) {
        curand_init(seed, 1, 0, state);
    }

    struct Scaling {
        dfloat3 vel;
        float* scaling_factor;
        Scaling(dfloat3 _vel, float* _sf) : vel(_vel), scaling_factor(_sf) {}
        __device__ void operator() (int idx) {
            auto sf = *scaling_factor;
            vel.x[idx] *= sf;
            vel.y[idx] *= sf;
            vel.z[idx] *= sf;
        }
    };
}

using namespace md::thermostats;

void BussiThermostat::init(State& state, unsigned long long seed) {
    this->dof = 3 * state.n_atoms;
    this->calculator = std::make_unique<KinEnergyCalculator>(state);

    init_curand<<<1, 1>>>(
        this->curand_state, 
        seed
    );
}

void BussiThermostat::stepTwo(State& state) {
    this->scheduler->get_temperature(state);    
    calculator->calc_kinetic_energy(state);

    // スケーリング要素を計算
    calc_scaling_factor<<<1, 1, 0, state.stream>>>(
        state.kinetic_energy, 
        this->scaling_factor, 
        this->curand_state, 
        this->dof, 
        1.0f / tau, 
        boltzmann_constant, 
        state.dt
    );

    // スケーリング
    thrust::for_each(
        thrust::cuda::par_nosync.on(state.stream), 
        thrust::make_counting_iterator<size_t>(0), 
        thrust::make_counting_iterator<size_t>(state.n_atoms), 
        Scaling(
            state.vel, 
            this->scaling_factor
        )
    ); 
}