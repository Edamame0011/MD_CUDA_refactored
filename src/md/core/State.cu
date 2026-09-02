#include <md/core/State.hpp>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

namespace md {
    State::State(int N) {
        this->n_atoms = N;

        // メモリの確保
        float *pos_, *vel_, *force_;
        cudaMalloc(&pos_, 3 * N * sizeof(float));
        this->pos.x = pos_;
        this->pos.y = pos_ + N;
        this->pos.z = pos_ + N + N;
        cudaMalloc(&vel_, 3 * N * sizeof(float));
        this->vel.x = vel_;
        this->vel.y = vel_ + N;
        this->vel.z = vel_ + N + N;
        cudaMalloc(&force_, 3 * N * sizeof(float));
        this->force.x = force_;
        this->force.y = force_ + N;
        this->force.z = force_ + N + N;

        cudaMalloc(&this->image.x, N * sizeof(int));
        cudaMalloc(&this->image.y, N * sizeof(int));
        cudaMalloc(&this->image.z, N * sizeof(int));

        cudaMalloc(&this->mass, N * sizeof(float));
        cudaMalloc(&this->mass_inv, N * sizeof(float));
        cudaMalloc(&this->species, N * sizeof(int));
        cudaMalloc(&this->particle_id, N * sizeof(int));

        // バッファの確保
        float *pos_buffer_, *vel_buffer_;
        cudaMalloc(&pos_buffer_, 3 * N * sizeof(float));
        this->pos_buffer.x = pos_buffer_;
        this->pos_buffer.y = pos_buffer_ + N;
        this->pos_buffer.z = pos_buffer_ + N + N;
        cudaMalloc(&vel_buffer_, 3 * N * sizeof(float));
        this->vel_buffer.x = vel_buffer_;
        this->vel_buffer.y = vel_buffer_ + N;
        this->vel_buffer.z = vel_buffer_ + N + N;

        cudaMalloc(&this->mass_buffer, N * sizeof(float));
        cudaMalloc(&this->mass_inv_buffer, N * sizeof(float));
        cudaMalloc(&this->species_buffer, N * sizeof(int));
        cudaMalloc(&this->particle_id_buffer, N * sizeof(int));
    }

    State::~State() {
        cudaFree(pos.x);
        cudaFree(vel.x);
        cudaFree(force.x);
        cudaFree(image.x);
        cudaFree(image.y);
        cudaFree(image.z);
        cudaFree(mass);
        cudaFree(mass_inv);
        cudaFree(species);
        cudaFree(particle_id);
        cudaFree(pos_buffer.x);
        cudaFree(vel_buffer.x);
        cudaFree(mass_buffer);
        cudaFree(mass_inv_buffer);
        cudaFree(species_buffer);
        cudaFree(particle_id_buffer);
    }

    void State::init(
        const float *h_pos_x, const float *h_pos_y, const float *h_pos_z, 
        const float *h_vel_x, const float *h_vel_y, const float *h_vel_z, 
        const float *h_force_x, const float *h_force_y, const float *h_force_z, 
        const float *h_mass, const int *h_species
    ) {
        // データの転送
        cudaMemcpy(this->pos.x, h_pos_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->pos.y, h_pos_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->pos.z, h_pos_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->vel.x, h_vel_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->vel.y, h_vel_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->vel.z, h_vel_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->force.x, h_force_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->force.y, h_force_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->force.z, h_force_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->mass, h_mass, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->species, h_species, n_atoms * sizeof(int), cudaMemcpyHostToDevice);

        // mass_inv = 1 / mass
        thrust::transform(
            thrust::device, 
            mass, 
            mass + n_atoms, 
            mass_inv, 
            [] __device__ (float mass) {
                return 1.0f / mass;
            } 
        );
    }

    void State::copy_vel(const float *h_vel_x, const float *h_vel_y, const float *h_vel_z) {
        cudaMemcpy(this->vel.x, h_vel_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->vel.y, h_vel_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->vel.z, h_vel_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
    }

    void State::swap_buffer() {
        std::swap(pos.x, pos_buffer.x);
        std::swap(pos.y, pos_buffer.y);
        std::swap(pos.z, pos_buffer.z);
        std::swap(vel.x, vel_buffer.x);
        std::swap(vel.y, vel_buffer.y);
        std::swap(vel.z, vel_buffer.z);
        std::swap(mass, mass_buffer);
        std::swap(mass_inv, mass_inv_buffer);
    }

    SimState::SimState() {
        cudaStreamCreate(&stream);
    }

    SimState::~SimState() {
        cudaStreamDestroy(stream);
    }
}