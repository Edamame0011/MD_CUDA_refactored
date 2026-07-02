#pragma once

#include <md/core/State.cuh>

namespace md {
    class Cell;

    class SortedCellList {
        public:
            SortedCellList(std::array<int, 3> _M, Cell* _cell, State& state);
            ~SortedCellList();
            void generate(State& state, bool* flag);
            void sort(State& state, bool* flag);

            int* get_cell_id() { return sorted_cell_id; }
            int* get_particle_id() { return sorted_particle_id; }
            dfloat3 get_sorted_pos() { return sorted_pos; }
            int* get_cell_start_idx() { return cell_start_idx; }
            std::array<int, 3> get_M() { return M; }

            SortedCellList(const SortedCellList&) = delete;
            SortedCellList& operator=(const SortedCellList&) = delete;
        private:
            std::array<int, 3> M; // 分割数
            std::array<float, 3> cell_size;
            int num_cells;
            
            int *cell_id, *sorted_cell_id, *particle_id, *sorted_particle_id;
            dfloat3 sorted_pos;
            int* cell_start_idx;
        
            // cub用のバッファとそのサイズ
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            // カーネル呼び出しの際のスレッド数
            int calc_cell_id_num_threads;
            int apply_sort_num_threads;

            Cell* cell;
    };
}