#include <md/core/State.hpp>

#include <md/core/constant.h>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

using DeviceVec3 = md::DeviceVec3;
using DeviceInt3 = md::DeviceInt3;

namespace {
    __global__ void reorder_kernel(
        const int* __restrict__ perm, 
        const DeviceVec3 pos, 
        const DeviceVec3 vel, 
        const DeviceInt3 image,
        const int* __restrict__ species, 
        const int* __restrict__ particle_id, 
        DeviceVec3 pos_buffer, 
        DeviceVec3 vel_buffer, 
        DeviceInt3 image_buffer,
        int* __restrict__ species_buffer, 
        int* __restrict__ particle_id_buffer, 
        const int num_atoms
    ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto old_idx = perm[idx];

        pos_buffer.x[idx] = pos.x[old_idx];
        pos_buffer.y[idx] = pos.y[old_idx];
        pos_buffer.z[idx] = pos.z[old_idx];
        vel_buffer.x[idx] = vel.x[old_idx];
        vel_buffer.y[idx] = vel.y[old_idx];
        vel_buffer.z[idx] = vel.z[old_idx];
        image_buffer.x[idx] = image.x[old_idx];
        image_buffer.y[idx] = image.y[old_idx];
        image_buffer.z[idx] = image.z[old_idx];
        species_buffer[idx] = species[old_idx];
        particle_id_buffer[idx] = particle_id[old_idx];
    }

    __global__ void reorder_normal_kernel(
        const int* __restrict__ perm, 
        const DeviceVec3 pos, 
        const DeviceVec3 vel, 
        const DeviceInt3 image, 
        const float* __restrict__ mass, 
        const float* __restrict__ mass_inv, 
        const int* __restrict__ species, 
        const int* __restrict__ particle_id, 
        const int* __restrict__ atomic_number, 
        DeviceVec3 pos_buffer, 
        DeviceVec3 vel_buffer, 
        DeviceInt3 image_buffer, 
        float* __restrict__ mass_buffer, 
        float* __restrict__ mass_inv_buffer, 
        int* __restrict__ species_buffer, 
        int* __restrict__ particle_id_buffer, 
        int* __restrict__ atomic_number_buffer, 
        const int num_atoms
    ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto old_idx = perm[idx];

        pos_buffer.x[idx] = pos.x[old_idx];
        pos_buffer.y[idx] = pos.y[old_idx];
        pos_buffer.z[idx] = pos.z[old_idx];
        vel_buffer.x[idx] = vel.x[old_idx];
        vel_buffer.y[idx] = vel.y[old_idx];
        vel_buffer.z[idx] = vel.z[old_idx];
        image_buffer.x[idx] = image.x[old_idx];
        image_buffer.y[idx] = image.y[old_idx];
        image_buffer.z[idx] = image.z[old_idx];
        mass_buffer[idx] = mass[old_idx];
        mass_inv_buffer[idx] = mass_inv[old_idx];
        atomic_number_buffer[idx] = atomic_number[old_idx];
        species_buffer[idx] = species[old_idx];
        particle_id_buffer[idx] = particle_id[old_idx];
    }
}

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

        cudaMalloc(&this->species, N * sizeof(int));
        cudaMalloc(&this->particle_id, N * sizeof(int));

        // バッファの確保
        float *pos_buffer_, *vel_buffer_, *force_buffer_;
        cudaMalloc(&pos_buffer_, 3 * N * sizeof(float));
        this->pos_buffer.x = pos_buffer_;
        this->pos_buffer.y = pos_buffer_ + N;
        this->pos_buffer.z = pos_buffer_ + N + N;
        cudaMalloc(&vel_buffer_, 3 * N * sizeof(float));
        this->vel_buffer.x = vel_buffer_;
        this->vel_buffer.y = vel_buffer_ + N;
        this->vel_buffer.z = vel_buffer_ + N + N;

        cudaMalloc(&force_buffer_, 3 * N * sizeof(float));
        this->force_buffer.x = force_buffer_;
        this->force_buffer.y = force_buffer_ + N;
        this->force_buffer.z = force_buffer_ + N + N;

        cudaMalloc(&this->image_buffer.x, N * sizeof(int));
        cudaMalloc(&this->image_buffer.y, N * sizeof(int));
        cudaMalloc(&this->image_buffer.z, N * sizeof(int));

        cudaMalloc(&this->species_buffer, N * sizeof(int));
        cudaMalloc(&this->particle_id_buffer, N * sizeof(int));

        cudaMalloc(&mass, N * sizeof(float));
        cudaMalloc(&mass_inv, N * sizeof(float));
        cudaMalloc(&atomic_number, N * sizeof(int));
    }

    State::~State() {
        cudaFree(pos.x);
        cudaFree(vel.x);
        cudaFree(force.x);
        cudaFree(image.x);
        cudaFree(image.y);
        cudaFree(image.z);
        cudaFree(species);
        cudaFree(particle_id);
        cudaFree(pos_buffer.x);
        cudaFree(vel_buffer.x);
        cudaFree(force_buffer.x);
        cudaFree(image_buffer.x);
        cudaFree(image_buffer.y);
        cudaFree(image_buffer.z);
        cudaFree(species_buffer);
        cudaFree(particle_id_buffer);
        cudaFree(mass);
        cudaFree(mass_inv);
        cudaFree(atomic_number);
    }

    void State::init(
        const float *h_pos_x, const float *h_pos_y, const float *h_pos_z, 
        const float *h_vel_x, const float *h_vel_y, const float *h_vel_z, 
        const float *h_force_x, const float *h_force_y, const float *h_force_z, 
        const int *h_species
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
        cudaMemcpy(this->species, h_species, n_atoms * sizeof(int), cudaMemcpyHostToDevice);

        thrust::sequence(
            thrust::device, 
            particle_id, 
            particle_id + n_atoms
        );

        cudaMemset(image.x, 0, n_atoms * sizeof(int));
        cudaMemset(image.y, 0, n_atoms * sizeof(int));
        cudaMemset(image.z, 0, n_atoms * sizeof(int));

        thrust::fill(thrust::device, mass, mass + n_atoms, 1.0f);
        thrust::fill(thrust::device, mass_inv, mass_inv + n_atoms, 1.0f);
    }

    void State::copy_vel(const float *h_vel_x, const float *h_vel_y, const float *h_vel_z) {
        cudaMemcpy(this->vel.x, h_vel_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->vel.y, h_vel_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->vel.z, h_vel_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
    }

    void State::swap_buffer() {
        std::swap(pos, pos_buffer);
        std::swap(vel, vel_buffer);
        std::swap(species, species_buffer);
        std::swap(particle_id, particle_id_buffer);
        std::swap(image, image_buffer);
    }

    void State::reorder(const int* perm, cudaStream_t stream) {
        const int num_blocks = (n_atoms + NUM_THREADS - 1) / NUM_THREADS;

        reorder_kernel<<<num_blocks, NUM_THREADS, 0, stream>>>(
            perm, 
            this->pos, 
            this->vel, 
            this->image, 
            this->species, 
            this->particle_id, 
            this->pos_buffer, 
            this->vel_buffer, 
            this->image_buffer, 
            this->species_buffer, 
            this->particle_id_buffer, 
            this->n_atoms
        );
    }

    StateNormal::StateNormal(int N) : State(N) {
        cudaMalloc(&mass_buffer, N * sizeof(float));
        cudaMalloc(&mass_inv_buffer, N * sizeof(float));
        cudaMalloc(&atomic_number_buffer, N * sizeof(int));
    }

    StateNormal::~StateNormal() {
        cudaFree(mass_buffer);
        cudaFree(mass_inv_buffer);
        cudaFree(atomic_number_buffer);
    }

    void StateNormal::init_normal(
        const float *h_pos_x, const float *h_pos_y, const float *h_pos_z, 
        const float *h_vel_x, const float *h_vel_y, const float *h_vel_z, 
        const float *h_force_x, const float *h_force_y, const float *h_force_z, 
        const int *h_species, const float *h_mass, const int* h_atomic_number
    ) {
        State::init(
            h_pos_x, h_pos_y, h_pos_z, 
            h_vel_x, h_vel_y, h_vel_z, 
            h_force_x, h_force_y, h_force_z, 
            h_species  
        );

        cudaMemcpy(this->mass, h_mass, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->atomic_number, h_atomic_number, n_atoms * sizeof(int), cudaMemcpyHostToDevice);
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

    void StateNormal::swap_buffer() {
        State::swap_buffer();
        std::swap(mass, mass_buffer);
        std::swap(mass_inv, mass_inv_buffer);
        std::swap(atomic_number, atomic_number_buffer);
    }

    void StateNormal::reorder(const int* perm, cudaStream_t stream) {
        const int num_blocks = (n_atoms + NUM_THREADS - 1) / NUM_THREADS;

        reorder_normal_kernel<<<num_blocks, NUM_THREADS, 0, stream>>>(
            perm, 
            this->pos, 
            this->vel, 
            this->image, 
            this->mass, 
            this->mass_inv, 
            this->species, 
            this->particle_id, 
            this->atomic_number, 
            this->pos_buffer, 
            this->vel_buffer, 
            this->image_buffer, 
            this->mass_buffer, 
            this->mass_inv_buffer, 
            this->species_buffer, 
            this->particle_id_buffer, 
            this->atomic_number_buffer, 
            this->n_atoms
        );
    }

    SimState::SimState() {
        cudaStreamCreate(&stream);
    }

    SimState::~SimState() {
        cudaStreamDestroy(stream);
    }
}
