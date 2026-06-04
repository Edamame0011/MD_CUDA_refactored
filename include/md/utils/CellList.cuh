// 立方体を想定して書いています。

#pragma once

#include <md/core/State.cuh>

namespace md {
    class CellList {
        public:
            CellList(int _M, float _Lbox, State& state);
            ~CellList();
            void generate(State& state, bool* flag);
            void sort(State& state, bool* flag);

            unsigned int* get_cell_id() { return sorted_cell_id; }
            unsigned int* get_particle_id() { return sorted_particle_id; }
            dfloat3 get_sorted_pos() { return sorted_pos; }
            unsigned int* get_cell_start_idx() { return cell_start_idx; }
            int get_M() { return M; }

            CellList(const CellList&) = delete;
            CellList& operator=(const CellList&) = delete;
        private:
            unsigned int M; // 分割数
            float cell_size;
            float Lbox;
            
            unsigned int *cell_id, *sorted_cell_id, *particle_id, *sorted_particle_id;
            dfloat3 sorted_pos;
            unsigned int* cell_start_idx;
        
            // cub用のバッファとそのサイズ
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            // カーネル呼び出しの際のスレッド数
            int calc_cell_id_num_threads;
            int apply_sort_num_threads;
    };
}