#include <md/cells/Cell.cuh>
#include <md/core/State.cuh>

namespace {
    __global__ void apply_pbc_kernel(
        dfloat3 pos, 
        dint3 box, 
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

        box.x[idx] += (int)shift_x;
        box.y[idx] += (int)shift_y;
        box.z[idx] += (int)shift_z;

        pos.x[idx] = px;
        pos.y[idx] = py;
        pos.z[idx] = pz;
    }
}

namespace md {
    void Cell::apply_pbc(State& state) const {
        auto N = state.n_atoms;

        int num_threads = 256;
        int num_blocks = (num_threads + N - 1) / num_threads;
        apply_pbc_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
            state.pos, 
            state.box, 
            lattice, 
            lattice_inv, 
            N
        );
    }
}