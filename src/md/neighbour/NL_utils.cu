#include <md/neighbour/NL_utils.cuh>

namespace md::neighbour {
    __global__ void check_top2(
        Top2* top2, 
        bool* flag, 
        float margin_sq
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            Top2 t = *top2;
            if (t.max1 + t.max2 + 2 * sqrtf(t.max1 * t.max2) > margin_sq) *flag = true;
        }
    }

    __global__ void update_nl_conf_kernel(
        const bool* flag, 
        const DeviceVec3 pos, 
        DeviceVec3 nl_conf, 
        int num_atoms
    ) {
        if (!*flag) return;
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < num_atoms) {
            nl_conf.x[idx] = pos.x[idx];
            nl_conf.y[idx] = pos.y[idx];
            nl_conf.z[idx] = pos.z[idx];
        }
    }
}