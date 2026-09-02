#include <md/thermostats/BussiThermostat.cuh>

#include <md/core/State.hpp>
#include <md/core/constant.h>
#include <md/temperature_schedulers/TemperatureScheduler.hpp>
#include <md/thermostats/KinEnergyCalculator.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace {
    __global__ void initialize_random_state(curandState* state, unsigned long long seed) {
        curand_init(seed, 0, 0, state);
    }
    
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
            if (u < 1.0f - 0.0331f * x_sq * x_sq || __logf(u) < 0.5f * x_sq + d * (1.0f - v + __logf(v))) {
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
        float f = __expf(-dt * tau_inv);
        float alpha2 = f + (targ_kin * (1.0f - f) * (r1 * r1 + r2 * r2 + 2 * g)) / (dof * current_kin) + 2.0f * r1 * sqrtf((targ_kin * f * (1.0f - f)) / (dof * current_kin));

        *scaling_factor = sqrtf(alpha2);
    }

    __global__ void scale_velocities(
        md::DeviceVec3 velocity, int atom_count, const float* scaling_factor) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= atom_count) {
            return;
        }

        const float scale = *scaling_factor;
        velocity.x[index] *= scale;
        velocity.y[index] *= scale;
        velocity.z[index] *= scale;
    }
}

namespace md::thermostats {
    BussiThermostat::BussiThermostat(float tau, TemperatureScheduler* scheduler)
        : tau_(tau), scheduler_(scheduler) {
        if (!std::isfinite(tau_) || tau_ <= 0.0f) {
            throw std::invalid_argument("Bussi thermostat tau must be finite and positive");
        }
        if (scheduler_ == nullptr) {
            throw std::invalid_argument("Bussi thermostat requires a temperature scheduler");
        }

        cudaMalloc(&scaling_factor_, sizeof(float));
        cudaMalloc(&curand_state_, sizeof(curandState));
    }

    BussiThermostat::~BussiThermostat() {
        cudaFree(curand_state_);
        cudaFree(scaling_factor_);
    }

    void BussiThermostat::init(State& state, SimState& simstate, unsigned long long seed) {
        this->atom_count_ = state.n_atoms;
        this->degrees_of_freedom_ = 3 * atom_count_;
        this->calculator_ = std::make_unique<KinEnergyCalculator>(state);

        initialize_random_state<<<1, 1, 0, simstate.stream>>>(curand_state_, seed);
    }

    void BussiThermostat::stepOne(State&, SimState&) {
        // 何もしない
    }

    void BussiThermostat::stepTwo(State& state, SimState& simstate) {
        scheduler_->get_temperature(state, simstate);
        calculator_->calc_kinetic_energy(state, simstate);

        calc_scaling_factor<<<1, 1, 0, simstate.stream>>>(
            calculator_->device_kinetic_energy(),
            scaling_factor_,
            curand_state_, 
            degrees_of_freedom_, 
            1.0f / tau_, 
            boltzmann_constant, 
            simstate.dt
        );

        const int blocks = (atom_count_ + NUM_THREADS - 1) / NUM_THREADS;
        scale_velocities<<<blocks, NUM_THREADS, 0, simstate.stream>>>(
            state.vel, 
            atom_count_, 
            scaling_factor_
        );
    }
}
