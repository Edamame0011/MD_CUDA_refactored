#include <md/cells/CubicCell.cuh>

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
    // zip
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            state.d_positions.x.begin(), 
            state.d_positions.y.begin(), 
            state.d_positions.z.begin(), 
            state.d_box.x.begin(), 
            state.d_box.y.begin(), 
            state.d_box.z.begin()
        )
    );

    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(
            state.d_positions.x.end(), 
            state.d_positions.y.end(), 
            state.d_positions.z.end(), 
            state.d_box.x.end(), 
            state.d_box.y.end(), 
            state.d_box.z.end()
        )
    );

    // 更新
    thrust::for_each(zip_begin, zip_end, ApplyPBC(Lbox));
}