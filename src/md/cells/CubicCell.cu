#include <md/cells/CubicCell.cuh>

#include <md/core/State.cuh>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
    __global__ void apply_pbc_kernel(
        dfloat3 pos, 
        dint3 box, 
        float Lbox, 
        int N
    ) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N) return;

        float Linv = 1.0f / Lbox;

        auto px = pos.x[idx];
        auto py = pos.y[idx];
        auto pz = pos.z[idx];

        float shift_x = floorf(px * Linv + 0.5f);
        float shift_y = floorf(py * Linv + 0.5f);
        float shift_z = floorf(pz * Linv + 0.5f);

        px -= Lbox * shift_x;
        py -= Lbox * shift_y;
        pz -= Lbox * shift_z;

        box.x[idx] += (int)shift_x;
        box.y[idx] += (int)shift_y;
        box.z[idx] += (int)shift_z;

        pos.x[idx] = px;
        pos.y[idx] = py;
        pos.z[idx] = pz;
    }

    __device__ void apply_pbc_elements_kernel(float* x, float* y, float* z, float* lattice) {
        float Lbox = lattice[0];
        float Linv = 1.0f / Lbox;

        *x -= Lbox * floorf(*x * Linv + 0.5f);
        *y -= Lbox * floorf(*y * Linv + 0.5f);
        *z -= Lbox * floorf(*z * Linv + 0.5f);   
    }

    __device__ void (*p) (float*, float*, float*, float*) = apply_pbc_elements_kernel;
}

using namespace md::cells;

using LatticeArr = std::array<std::array<float, 3>, 3>;
using FuncPtr = void (*) (float*, float*, float*, float*);

CubicCell::CubicCell(LatticeArr& _lattice) : Cell(_lattice) {
    cudaMemcpyFromSymbol(&apply_pbc_ptr, p, sizeof(FuncPtr));
    cudaMemcpy(d_lattice, lattice.data(), 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void CubicCell::apply_pbc(State& state) const {
    auto N = state.n_atoms;
    float Lbox = lattice[0][0];

    int num_threads = 256;
    int num_blocks = (num_threads + N - 1) / num_threads;
    apply_pbc_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        state.pos, 
        state.box, 
        Lbox, 
        N
    );
}