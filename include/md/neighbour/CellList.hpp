#pragma once

#include <array>
#include <thrust/device_vector.h>

namespace md {
    class State;
    class SimState;
    class Cell;

    class CellList {
        public:
            CellList(std::array<int, 3> _M, State& state, Cell& cell);
            ~CellList();
            void generate(State& state, SimState& simstate, Cell& cell, bool* flag);
            void sort(State& state, SimState& simstate, bool* flag);

            int* get_cell_id() { return thrust::raw_pointer_cast(sorted_cell_id.data()); }
            int* get_perm() { return thrust::raw_pointer_cast(sorted_perm.data()); }
            int* get_cell_start_idx() { return thrust::raw_pointer_cast(cell_start_idx.data()); }
            int get_num_cells() const { return num_cells; }
            std::array<int, 3> get_M() const { return M; }

            CellList(const CellList&) = delete;
            CellList& operator=(const CellList&) = delete;
        private:
            std::array<int, 3> M;
            std::array<float, 3> cell_size;
            int num_cells;

            thrust::device_vector<int> cell_id, perm, sorted_cell_id, sorted_perm, cell_start_idx;

            // cub用のバッファとそのサイズ
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
    };
}