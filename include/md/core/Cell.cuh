#pragma once

#include <array>

namespace md{
    class State;
    class SimState;

    class Cell {
        public:
            Cell(std::array<float, 3> _lattice): 
            lattice{_lattice[0], _lattice[1], _lattice[2]}, 
            lattice_inv{1.0f/_lattice[0], 1.0f/_lattice[1], 1.0f/_lattice[2]} {}

            __forceinline__ __device__ void apply_pbc_device(float* x, float* y, float* z) const {
                *x -= lattice.x * roundf(*x * lattice_inv.x);
                *y -= lattice.y * roundf(*y * lattice_inv.y);
                *z -= lattice.z * roundf(*z * lattice_inv.z);
            }

            __forceinline__ __device__ void apply_pbc_wrap_device(float* x, float* y, float* z) const {
                *x -= lattice.x * floorf(*x * lattice_inv.x);
                *y -= lattice.y * floorf(*y * lattice_inv.y);
                *z -= lattice.z * floorf(*z * lattice_inv.z);
            }

            void apply_pbc(State& state, SimState& simstate) const;

            std::array<float, 3> get_lattice() { return std::array<float, 3>{lattice.x, lattice.y, lattice.z}; }
        private:
            const float3 lattice;
            const float3 lattice_inv;
    };
}