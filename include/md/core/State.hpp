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
        int* species;           // 粒子種類 (通し番号)
        int* particle_id;       // 現在のインデックス -> ID

        // ソートのためのバッファ
        DeviceVec3 pos_buffer, vel_buffer, force_buffer;
        DeviceInt3 image_buffer;
        int *species_buffer, *particle_id_buffer;

        float *mass, *mass_inv;
        int *atomic_number;

        int n_atoms, n_species;

        State(int n_atoms, int n_species);
        virtual ~State();
        void init(
            const float *h_pos_x, const float *h_pos_y, const float *h_pos_z, 
            const float *h_vel_x, const float *h_vel_y, const float *h_vel_z, 
            const float *h_force_x, const float *h_force_y, const float *h_force_z, 
            const int *species
        );
        void copy_vel(const float *h_vel_x, const float *h_vel_y, const float *h_vel_z);

        virtual void reorder(const int* perm, cudaStream_t stream);
        virtual void swap_buffer();

        State(const State&) = delete;
        State& operator=(const State&) = delete;
    };

    struct StateNormal : public State {
        float *mass_buffer, *mass_inv_buffer;
        int *atomic_number_buffer;

        void reorder(const int* perm, cudaStream_t stream) override;
        void swap_buffer() override;
        void init_normal(
            const float *h_pos_x, const float *h_pos_y, const float *h_pos_z, 
            const float *h_vel_x, const float *h_vel_y, const float *h_vel_z, 
            const float *h_force_x, const float *h_force_y, const float *h_force_z, 
            const int *h_species, const float *h_mass, const int* atomic_number
        );

        StateNormal(int n_atoms, int n_species);
        ~StateNormal();
    };

    struct SimState {
        float dt = 0.0f;
        int current_steps = 0;
        cudaStream_t stream;

        SimState();
        ~SimState();

        SimState(const SimState&) = delete;
        SimState& operator=(const SimState&) = delete;
    };
}
