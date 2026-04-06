#ifndef MD_STATE_CUH
#define MD_STATE_CUH

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

struct dfloat3 {
    float *x, *y, *z;
};

struct dint3 {
    int *x, *y, *z;
};

namespace md {
    struct StateView {
        dfloat3 pos, vel, force;
        dint3 box;
        float* mass;
        float* mass_inv;
        int* atomic_numbers;
    
        float* kinetic_energy;
    };

    class State {
        StateView view;

        public:
            int n_atoms = 0;
            float dt = 0.0f;
            int current_steps = 0;
            float potential_energy = 0;

            cudaStream_t stream;

            State(int N) {
                cudaMalloc(&view.pos.x, N * sizeof(float));
                cudaMalloc(&view.pos.y, N * sizeof(float));
                cudaMalloc(&view.pos.z, N * sizeof(float));
                cudaMalloc(&view.vel.x, N * sizeof(float));
                cudaMalloc(&view.vel.y, N * sizeof(float));
                cudaMalloc(&view.vel.z, N * sizeof(float));
                cudaMalloc(&view.force.x, N * sizeof(float));
                cudaMalloc(&view.force.y, N * sizeof(float));
                cudaMalloc(&view.force.z, N * sizeof(float));
                cudaMalloc(&view.box.x, N * sizeof(int));
                cudaMalloc(&view.box.y, N * sizeof(int));
                cudaMalloc(&view.box.z, N * sizeof(int));
                cudaMalloc(&view.mass, N * sizeof(float));
                cudaMalloc(&view.mass_inv, N * sizeof(float));
                cudaMalloc(&view.atomic_numbers, N * sizeof(int));
                cudaMalloc(&view.kinetic_energy, sizeof(float));
                cudaStreamCreate(&stream);
                this->n_atoms = N;

                cudaMemset(view.box.x, 0, N * sizeof(int));
                cudaMemset(view.box.y, 0, N * sizeof(int));
                cudaMemset(view.box.z, 0, N * sizeof(int));
            }
            ~State() {
                cudaFree(view.pos.x);
                cudaFree(view.pos.y);
                cudaFree(view.pos.z);
                cudaFree(view.vel.x);
                cudaFree(view.vel.y);
                cudaFree(view.vel.z);
                cudaFree(view.force.x);
                cudaFree(view.force.y);
                cudaFree(view.force.z);
                cudaFree(view.box.x);
                cudaFree(view.box.y);
                cudaFree(view.box.z);
                cudaFree(view.mass);
                cudaFree(view.mass_inv);
                cudaFree(view.atomic_numbers);
                cudaFree(view.kinetic_energy);
                cudaStreamDestroy(stream);
            }
            StateView get_view() {
                return this->view;
            }
            void copy(
                float *h_pos_x, float *h_pos_y, float *h_pos_z, 
                float *h_vel_x, float *h_vel_y, float *h_vel_z, 
                float *h_force_x, float *h_force_y, float *h_force_z, 
                float *h_mass, int *h_atomic_numbers
            ) {
                cudaMemcpy(view.pos.x, h_pos_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.pos.y, h_pos_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.pos.z, h_pos_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.vel.x, h_vel_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.vel.y, h_vel_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.vel.z, h_vel_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.force.x, h_force_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.force.y, h_force_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.force.z, h_force_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.mass, h_mass, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.atomic_numbers, h_atomic_numbers, n_atoms * sizeof(int), cudaMemcpyHostToDevice);
                thrust::transform(
                    thrust::device, 
                    view.mass, 
                    view.mass + n_atoms, 
                    view.mass_inv, 
                    [] __device__ (float mass) {
                        return 1.0f / mass;
                    }
                );
            }
            void copy_vel(float *h_vel_x, float *h_vel_y, float *h_vel_z) {
                cudaMemcpy(view.vel.x, h_vel_x, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.vel.y, h_vel_y, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(view.vel.z, h_vel_z, n_atoms * sizeof(float), cudaMemcpyHostToDevice);
            }

            State(const State&) = delete;
            State& operator=(const State&) = delete;
    };
}

#endif