#include <md/thermostats/KinEnergyCalculator.hpp>

#include <md/core/State.hpp>
#include <md/core/constant.h>

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace {
    void check_cuda(cudaError_t error, const char* operation) {
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
        }
    }

    struct ParticleKineticEnergy {
        md::DeviceVec3 velocity;
        const float* mass;
        float inverse_conversion_factor;

        __device__ float operator()(int index) const {
            const float vx = velocity.x[index];
            const float vy = velocity.y[index];
            const float vz = velocity.z[index];
            const float particle_mass = mass[index];
            return 0.5f * particle_mass * (vx * vx + vy * vy + vz * vz)
                * inverse_conversion_factor;
        }
    };
}

namespace md::thermostats {
    KinEnergyCalculator::KinEnergyCalculator(const State& state)
        : capacity_(state.n_atoms) {
        if (capacity_ <= 0) {
            throw std::invalid_argument("kinetic energy requires at least one atom");
        }
        if (!std::isfinite(conversion_factor) || conversion_factor <= 0.0f) {
            throw std::invalid_argument("conversion_factor must be finite and positive");
        }

        try {
            check_cuda(cudaMalloc(&d_kinetic_energy_, sizeof(*d_kinetic_energy_)),
                       "allocating float kinetic energy");
            const ParticleKineticEnergy transform{
                state.vel,
                state.mass,
                1.0f / conversion_factor
            };
            const auto input = thrust::make_transform_iterator(
                thrust::counting_iterator<int>(0), transform);
            check_cuda(cub::DeviceReduce::Sum(
                nullptr,
                temp_storage_bytes_,
                input,
                d_kinetic_energy_,
                capacity_
            ), "querying kinetic energy reduction storage");
            check_cuda(cudaMalloc(&d_temp_storage_, temp_storage_bytes_),
                       "allocating kinetic energy reduction storage");
        } catch (...) {
            cudaFree(d_temp_storage_);
            cudaFree(d_kinetic_energy_);
            throw;
        }
    }

    KinEnergyCalculator::~KinEnergyCalculator() {
        cudaFree(d_temp_storage_);
        cudaFree(d_kinetic_energy_);
    }

    void KinEnergyCalculator::calc_kinetic_energy(const State& state, SimState& simstate) {
        if (state.n_atoms != capacity_) {
            throw std::invalid_argument("State atom count differs from calculator capacity");
        }
        if (!std::isfinite(conversion_factor) || conversion_factor <= 0.0f) {
            throw std::invalid_argument("conversion_factor must be finite and positive");
        }

        const ParticleKineticEnergy transform{
            state.vel,
            state.mass,
            1.0f / conversion_factor
        };
        const auto input = thrust::make_transform_iterator(
            thrust::counting_iterator<int>(0), transform);

        check_cuda(cub::DeviceReduce::Sum(
            d_temp_storage_,
            temp_storage_bytes_,
            input,
            d_kinetic_energy_,
            state.n_atoms,
            simstate.stream
        ), "reducing kinetic energy");

    }

    float KinEnergyCalculator::calc_kinetic_energy_host(
        const State& state, SimState& simstate) {
        calc_kinetic_energy(state, simstate);

        float result = 0.0f;
        check_cuda(cudaMemcpyAsync(
            &result,
            d_kinetic_energy_,
            sizeof(result),
            cudaMemcpyDeviceToHost,
            simstate.stream
        ), "copying kinetic energy to host");
        check_cuda(cudaStreamSynchronize(simstate.stream),
                   "synchronizing kinetic energy stream");
        return result;
    }
}
