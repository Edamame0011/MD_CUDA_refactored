#pragma once

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

struct dfloat3 {
    float *x, *y, *z;
};

struct dint3 {
    int *x, *y, *z;
};

namespace md {
    class State {
        public:
            dfloat3 pos, vel, force;
            dint3 box;
            float* mass;
            float* mass_inv;
            int* atomic_numbers;
            float* kinetic_energy;

            int n_atoms = 0;
            float dt = 0.0f;
            int current_steps = 0;
            float potential_energy = 0;

            cudaStream_t stream;

            State(int N) {
                // pos・vel・forcesはx, y, zが並ぶように確保
                float *_pos, *_vel, *_force;
                cudaMalloc(&_pos, 3 * N * sizeof(float));
                pos.x = _pos;
                pos.y = _pos + N;
                pos.z = _pos + 2 * N;

                cudaMalloc(&_vel, 3 * N * sizeof(float));
                vel.x = _vel;
                vel.y = _vel + N;
                vel.z = _vel + 2 * N;

                cudaMalloc(&_force, 3 * N * sizeof(float));
                force.x = _force;
                force.y = _force + N;
                force.z = _force + 2 * N;

                cudaMalloc(&box.x, N * sizeof(int));
                cudaMalloc(&box.y, N * sizeof(int));
                cudaMalloc(&box.z, N * sizeof(int));
                cudaMalloc(&mass, N * sizeof(float));
                cudaMalloc(&mass_inv, N * sizeof(float));
                cudaMalloc(&atomic_numbers, N * sizeof(int));
                cudaMalloc(&kinetic_energy, sizeof(float));
                cudaStreamCreate(&stream);
                this->n_atoms = N;

                cudaMemset(box.x, 0, N * sizeof(int));
                cudaMemset(box.y, 0, N * sizeof(int));
                cudaMemset(box.z, 0, N * sizeof(int));
            }
            ~State() {
                cudaFree(pos.x);
                // cudaFree(pos.y);
                // cudaFree(pos.z);
                cudaFree(vel.x);
                // cudaFree(vel.y);
                // cudaFree(vel.z);
                cudaFree(force.x);
                // cudaFree(force.y);
                // cudaFree(force.z);
                cudaFree(box.x);
                cudaFree(box.y);
                cudaFree(box.z);
                cudaFree(mass);
                cudaFree(mass_inv);
                cudaFree(atomic_numbers);
                cudaFree(kinetic_energy);
                cudaStreamDestroy(stream);
            }
            void copy(
                const float *h_pos_x, const float *h_pos_y, const float *h_pos_z, 
                const float *h_vel_x, const float *h_vel_y, const float *h_vel_z, 
                const float *h_force_x, const float *h_force_y, const float *h_force_z, 
                const float *h_mass, const int *h_atomic_numbers
            ) {
                cudaMemcpy(pos.x, h_pos_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(pos.y, h_pos_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(pos.z, h_pos_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(vel.x, h_vel_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(vel.y, h_vel_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(vel.z, h_vel_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(force.x, h_force_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(force.y, h_force_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(force.z, h_force_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(mass, h_mass, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(atomic_numbers, h_atomic_numbers, n_atoms * sizeof(int), cudaMemcpyHostToDevice);
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
            void copy_vel(const float *h_vel_x, const float *h_vel_y, const float *h_vel_z) {
                cudaMemcpy(vel.x, h_vel_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(vel.y, h_vel_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(vel.z, h_vel_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
            }

            State(const State&) = delete;
            State& operator=(const State&) = delete;
    };
}