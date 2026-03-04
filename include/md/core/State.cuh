#ifndef STATE_CUH
#define STATE_CUH

#include <thrust/device_vector.h>

namespace md {
    struct State {
        // ====== デバイス側で保持する変数 ======
        struct device_float3 {
            thrust::device_vector<float> x, y, z;
        } d_positions, d_forces, d_velocities;              // 座標・力・速度
        struct device_int3 {
            thrust::device_vector<int> x, y, z;
        } d_box;                                            // 境界を超えた回数
        thrust::device_vector<float> d_masses;              // 質量
        thrust::device_vector<int64_t> d_atomic_numbers;    // 原子番号
    
        // ====== ホスト側で保持する変数 ======
        int n_atoms;                                        // 粒子数
        float dt;                                           // 時間刻み幅
        float potential_energy;                             // ポテンシャルエネルギー
        int current_steps;                                  // 現在のステップ数
    };
}

#endif