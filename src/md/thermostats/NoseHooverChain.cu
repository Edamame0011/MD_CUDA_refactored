#include <md/thermostats/NoseHooverChain.cuh>
#include <md/core/State.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>

#include <cmath>

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

NoseHooverChain::NoseHooverChain(const int _length, const float _tau, TemperatureScheduler *_scheduler)
 :  length(_length), 
    tau(_tau), 
    scheduler(_scheduler), 
    chain_forces(std::vector<float>(_length, 0.0f)), 
    chain_velocities(std::vector<float>(_length, 0.0f)), 
    chain_positions(std::vector<float>(_length, 0.0f)), 
    chain_masses(std::vector<float>(_length, 0.0f))
{
    if (length == 1) {
        this->update_step_one = [this] (State& state) {
            this->NHC1(state);
        };
        this->update_step_two = [this] (State& state) {
            this->NHC1(state);
        };
    }
    else if (length == 2) {
        this->update_step_one = [this] (State& state) {
            this->NHC2(state);
        };
        this->update_step_two = [this] (State& state) {
            this->NHC2(state);
        };
    }
    else {
        this->update_step_one = [this] (State& state) {
            this->NHCM(state);
        };
        this->update_step_two = [this] (State& state) {
            this->NHCM(state);
        };
    }

    this->dof = 0;
}

void NoseHooverChain::init(State& state) {
    this->dof = 3 * state.n_atoms;
}

void NoseHooverChain::stepOne(State& state) {
    this->update_step_one(state);
}

void NoseHooverChain::stepTwo(State& state) {
    this->update_step_two(state);
}

void NoseHooverChain::NHC1(State& state) {
    float AKIN = 2.0f * md::utils::compute::calc_kinetic_energy(state);
    float scaling_factor = 1.0f;
    float dt = state.dt;
    float target_temperature = this->scheduler->get_temperature(state.current_steps);

    this->set_masses(target_temperature);

    // 逆順の更新
    this->chain_forces[0] = (AKIN - (dof * target_temperature * boltzmann_constant)) / this->chain_masses[0];
    this->chain_velocities[0] += dt * 0.25f * this->chain_forces[0];

    // スケーリング
    scaling_factor *= std::exp(-0.5f * dt * this->chain_velocities[0]);
    AKIN *= std::exp(-dt * this->chain_velocities[0]);

    // 変異の更新
    this->chain_positions[0] += dt * 0.5f * this->chain_velocities[0];

    // 順方向の更新
    this->chain_forces[0] = (AKIN - (dof * target_temperature * boltzmann_constant)) / this->chain_masses[0];
    this->chain_velocities[0] += dt * 0.25f * this->chain_forces[0];

    // 速度のスケーリング
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
        Scaling(scaling_factor)
    );
}

void NoseHooverChain::NHC2(State& state) {
    float AKIN = 2.0f * md::utils::compute::calc_kinetic_energy(state);
    float scaling_factor = 1.0f;
    float dt = state.dt;
    float target_temperature = this->scheduler->get_temperature(state.current_steps);

    this->set_masses(target_temperature);

    // 逆順の更新
    this->chain_forces[1] = ((this->chain_masses[0] * this->chain_velocities[0] * this->chain_velocities[0]) - (boltzmann_constant * target_temperature)) / this->chain_masses[1];
    this->chain_velocities[1] += dt * 0.25f * this->chain_forces[1];
    this->chain_velocities[0] *= std::exp(-0.125f * dt * this->chain_velocities[1]);
    this->chain_forces[0] = (AKIN - (this->dof * target_temperature * boltzmann_constant)) / this->chain_masses[0];
    this->chain_velocities[0] += 0.25f * dt * this->chain_forces[0];
    this->chain_velocities[0] *= std::exp(-0.125f * dt * this->chain_velocities[1]);

    // スケーリング
    scaling_factor *= std::exp(-0.5f * dt * this->chain_velocities[0]);
    AKIN *= std::exp(-dt * this->chain_velocities[0]);

    // 変異の更新
    this->chain_positions[0] += dt * 0.5f * this->chain_velocities[0];
    this->chain_positions[1] += dt * 0.5f * this->chain_velocities[1];

    // 順方向の更新
    this->chain_velocities[0] *= std::exp(-0.125f * dt * this->chain_velocities[1]);
    this->chain_forces[0] = (AKIN - (this->dof * target_temperature * boltzmann_constant)) / this->chain_masses[0];
    this->chain_velocities[0] += 0.25f * dt * this->chain_forces[0];
    this->chain_velocities[0] *= std::exp(-0.125f * dt * this->chain_velocities[1]);
    this->chain_forces[1] = ((this->chain_masses[0] * this->chain_velocities[0] * this->chain_velocities[0]) - (boltzmann_constant * target_temperature)) / this->chain_masses[1];
    this->chain_velocities[1] += dt * 0.25f * this->chain_forces[1];

    // 速度のスケーリング
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
        Scaling(scaling_factor)
    );
}

void NoseHooverChain::NHCM(State& state) {
    float AKIN = 2.0f * md::utils::compute::calc_kinetic_energy(state);
    float scaling_factor = 1.0f;
    float dt = state.dt;
    float target_temperature = this->scheduler->get_temperature(state.current_steps);

    this->set_masses(target_temperature);

    // 逆順の更新
    this->chain_forces[length -1] = ((this->chain_masses[length - 2] * this->chain_velocities[length - 2] * this->chain_velocities[length - 2]) - (boltzmann_constant * target_temperature)) / this->chain_masses[length - 1];
    this->chain_velocities[length - 1] += dt * 0.25f * this->chain_forces[length - 1];
    for (int i = length - 2; i > 0; i --) {
        this->chain_velocities[i] *= std::exp(-this->chain_velocities[i + 1] * dt * 0.125f);
        this->chain_forces[i] = ((this->chain_masses[i - 1] * this->chain_velocities[i - 1] * this->chain_velocities[i - 1]) - (boltzmann_constant * target_temperature)) / this->chain_masses[i];
        this->chain_velocities[i] += dt * 0.25f * this->chain_forces[i];
        this->chain_velocities[i] *= std::exp(-this->chain_velocities[i + 1] * dt * 0.125f);
    }
    this->chain_velocities[0] *= std::exp(-this->chain_velocities[1] * dt * 0.125f);
    this->chain_forces[0] = (AKIN - (this->dof * boltzmann_constant * target_temperature)) / this->chain_masses[0];
    this->chain_velocities[0] += dt * 0.25f * this->chain_forces[0];
    this->chain_velocities[0] *= std::exp(-this->chain_velocities[1] * dt * 0.125f);

    // スケーリング
    scaling_factor *= std::exp(-0.5f * dt * this->chain_velocities[0]);
    AKIN *= std::exp(-dt * this->chain_velocities[0]);

    for (int i = 0; i < length; i ++) {
        this->chain_positions[i] += 0.5f * dt * this->chain_velocities[i];
    }

    // 順方向の更新
    this->chain_velocities[0] *= std::exp(-this->chain_velocities[1] * dt * 0.125f);
    this->chain_forces[0] = (AKIN - (this->dof * boltzmann_constant * target_temperature)) / this->chain_masses[0];
    this->chain_velocities[0] += dt * 0.25f * this->chain_forces[0];
    this->chain_velocities[0] *= std::exp(-this->chain_velocities[1] * dt * 0.125f);
    for (int i = 1; i < length - 1; i ++) {
        this->chain_velocities[i] *= std::exp(-this->chain_velocities[i + 1] * dt * 0.125f);
        this->chain_forces[i] = ((this->chain_masses[i - 1] * this->chain_velocities[i - 1] * this->chain_velocities[i - 1]) - (boltzmann_constant * target_temperature)) / this->chain_masses[i];
        this->chain_velocities[i] += dt * 0.25f * this->chain_forces[i];
        this->chain_velocities[i] *= std::exp(-this->chain_velocities[i + 1] * dt * 0.125f);
    }
    this->chain_forces[length -1] = ((this->chain_masses[length - 2] * this->chain_velocities[length - 2] * this->chain_velocities[length - 2]) - (boltzmann_constant * target_temperature)) / this->chain_masses[length - 1];
    this->chain_velocities[length - 1] += dt * 0.25f * this->chain_forces[length - 1];

    // 速度のスケーリング
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
        Scaling(scaling_factor)
    );
}

void NoseHooverChain::set_masses(float target_temperature) {
    float tau2 = tau * tau;
    std::fill(this->chain_masses.begin(), this->chain_masses.end(), boltzmann_constant * target_temperature * tau2);
    this->chain_masses[0] *= this->dof;
}