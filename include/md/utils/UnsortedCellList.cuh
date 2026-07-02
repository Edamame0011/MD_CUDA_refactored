#pragma once

#include <md/core/State.cuh>

namespace md {
    class Cell;

    class UnsortedCellList {
        public:
            UnsortedCellList(std::array<int, 3> _M, Cell* _cell, State& state);
            ~UnsortedCellList();
            void generate(State& state, bool* flag);

            int* get_head() { return head; }
            int* get_next() { return next; }
            std::array<int, 3> get_M() { return M; }
            std::array<float, 3> get_cell_size() { return cell_size; }

            UnsortedCellList(const UnsortedCellList&) = delete;
            UnsortedCellList& operator=(const UnsortedCellList&) = delete;
        private:
            std::array<int, 3> M; // 分割数
            std::array<float, 3> cell_size;
            int num_cells;
            
            int* head;
            int* next;

            // カーネル呼び出しの際のスレッド数
            int calc_cell_id_num_threads;
            int apply_sort_num_threads;

            Cell* cell;
    };
}