#include <md/cells/CubicCell.cuh>
#include <thrust/execution_policy.h>

namespace {
    struct ApplyPBC {
        float Lbox;
        ApplyPBC(float _Lbox) : Lbox(_Lbox) {}
        template <typename Tuple>
        __host__ __device__ void operator() (Tuple t) {
            float Linv = 1.0f / Lbox;

            float shift_x = floorf(thrust::get<0>(t) * Linv + 0.5f);
            float shift_y = floorf(thrust::get<1>(t) * Linv + 0.5f);
            float shift_z = floorf(thrust::get<2>(t) * Linv + 0.5f);

            thrust::get<0>(t) -= Lbox * shift_x;
            thrust::get<1>(t) -= Lbox * shift_y;
            thrust::get<2>(t) -= Lbox * shift_z;

            thrust::get<3>(t) += (int)shift_x;
            thrust::get<4>(t) += (int)shift_y;
            thrust::get<5>(t) += (int)shift_z;
        }
    };
}

using namespace md::cells;

void CubicCell::apply_pbc(State& state) const {
    auto view = state.get_view();
    auto N = state.n_atoms;

    // 更新
    thrust::for_each(
        thrust::device, 
        thrust::make_zip_iterator(thrust::make_tuple(
            view.pos.x, 
            view.pos.y, 
            view.pos.z, 
            view.box.x, 
            view.box.y, 
            view.box.z
        )), 
        thrust::make_zip_iterator(thrust::make_tuple(
            view.pos.x + N, 
            view.pos.y + N, 
            view.pos.z + N, 
            view.box.x + N, 
            view.box.y + N, 
            view.box.z + N
        )), 
        ApplyPBC(Lbox)
    );
}