#pragma once;

#include <md/core/State.hpp>
#include <md/core/Cell.cuh>

namespace md::neighbour {
    struct Top2 {
        float max1, max2;
        
        __host__ __device__ Top2() : max1(0.0f), max2(0.0f) {}
        __host__ __device__ Top2(float m1) : max1(m1), max2(0.0f) {}
        __host__ __device__ Top2(float m1, float m2) : max1(m1), max2(m2) {}
    };

    __global__ void check_top2(
        Top2* top2, 
        bool* flag, 
        float margin_sq
    );

    __global__ void update_nl_conf_kernel(
        const bool* flag, 
        const DeviceVec3 pos, 
        DeviceVec3 nl_conf, 
        int num_atoms
    );

    struct CalcDist {
        DeviceVec3 pos;
        DeviceVec3 nl_conf;
        Cell cell;

        CalcDist(
            DeviceVec3 _pos, 
            DeviceVec3 _nl_conf, 
            Cell _cell
        ) : pos(_pos), nl_conf(_nl_conf), cell(_cell) {}

        __device__ Top2 operator () (const int idx) const {
            auto dx = pos.x[idx] - nl_conf.x[idx];
            auto dy = pos.y[idx] - nl_conf.y[idx];
            auto dz = pos.z[idx] - nl_conf.z[idx];

            // PBC補正
            cell.apply_pbc_device(&dx, &dy, &dz);

            float dist_sq = dx * dx + dy * dy + dz * dz;

            return Top2(dist_sq);
        }
    };

    // 2つのTop2オブジェクトから新たな一つのTop2オブジェクトを作成
    struct MergeTop2 {
        __host__ __device__ Top2 operator () (const Top2& a, const Top2& b) const {
            float max1 = fmaxf(a.max1, b.max1);
            float max2 = fmaxf(fminf(a.max1, b.max1), fmaxf(a.max2, b.max2));
            return Top2(max1, max2);        
        }
    };
};