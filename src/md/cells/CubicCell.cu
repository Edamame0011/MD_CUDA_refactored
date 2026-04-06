#include <md/cells/CubicCell.cuh>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
    struct ApplyPBC {
        dfloat3 pos;
        dint3 box;
        float Lbox;
        ApplyPBC(
            dfloat3 _pos, 
            dint3 _box, 
            float _Lbox
        ) : pos(_pos), 
            box(_box), 
            Lbox(_Lbox) {}
        __host__ __device__ void operator() (int idx) {
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
    };
}

using namespace md::cells;

void CubicCell::apply_pbc(State& state) const {
    auto view = state.get_view();
    auto N = state.n_atoms;

    // 更新
    thrust::for_each(
        thrust::cuda::par_nosync.on(state.stream), 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        ApplyPBC(
            view.pos, 
            view.box, 
            Lbox
        )
    );
}