#pragma once

#include <cuda_runtime.h>

namespace md {
    struct DeviceVec3 {
        float *x, *y, *z;
    };

    struct DeviceInt3 {
        int *x, *y, *z; 
    };

    struct State {
        DeviceVec3 pos, vel, force;
        DeviceInt3 image;
        float* mass;
        float* mass_inv;
        int* species;           // 粒子種類
        int* particle_id;       // 現在のインデックス -> ID

        // ソートのためのバッファ
        DeviceVec3 pos_buffer, vel_buffer;
        DeviceInt3 image_buffer;
        float *mass_buffer, *mass_inv_buffer;
        int *species_buffer, *particle_id_buffer;

        int n_atoms;

        State(int N);
        ~State();
        void init(
            const float *h_pos_x, const float *h_pos_y, const float *h_pos_z, 
            const float *h_vel_x, const float *h_vel_y, const float *h_vel_z, 
            const float *h_force_x, const float *h_force_y, const float *h_force_z, 
            const float *h_mass, const int *species
        );
        void copy_vel(const float *h_vel_x, const float *h_vel_y, const float *h_vel_z);
        void swap_buffer();

        State(const State&) = delete;
        State& operator=(const State&) = delete;
    };

    struct SimState {
        float dt = 0.0f;
        int current_steps = 0;
        cudaStream_t stream;

        SimState();
        ~SimState();
    };
}