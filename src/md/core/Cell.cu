#include <md/core/Cell.cuh>
#include <md/core/State.hpp>

#include <thrust/fill.h>
#include <thrust/execution_policy.h>

using DeviceVec3 = md::DeviceVec3;
using DeviceInt3 = md::DeviceInt3;

namespace {
    __global__ void apply_pbc_kernel(
        DeviceVec3 pos, 
        DeviceInt3 image, 
        const float3 lattice, 
        const float3 lattice_inv, 
        const int N
    ) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N) return;

        auto px = pos.x[idx];
        auto py = pos.y[idx];
        auto pz = pos.z[idx];

        float shift_x = roundf(px * lattice_inv.x);
        float shift_y = roundf(py * lattice_inv.y);
        float shift_z = roundf(pz * lattice_inv.z);

        px -= lattice.x * shift_x;
        py -= lattice.y * shift_y;
        pz -= lattice.z * shift_z;

        image.x[idx] += (int)shift_x;
        image.y[idx] += (int)shift_y;
        image.z[idx] += (int)shift_z;

        pos.x[idx] = px;
        pos.y[idx] = py;
        pos.z[idx] = pz;
    }
}

namespace md {
    Cell::Cell(int N, std::array<float, 3> _lattice) 
    : lattice{_lattice[0], _lattice[1], _lattice[2]}, lattice_inv{1.0f/_lattice[0], 1.0f/_lattice[1], 1.0f/_lattice[2]} {}

    void Cell::apply_pbc(State& state, SimState& simstate) const {
        auto N = state.n_atoms;

        int num_threads = 256;
        int num_blocks = (num_threads + N - 1) / num_threads;
        apply_pbc_kernel<<<num_blocks, num_threads, 0, simstate.stream>>>(
            state.pos, 
            state.image, 
            lattice, 
            lattice_inv, 
            N
        );
    }
}