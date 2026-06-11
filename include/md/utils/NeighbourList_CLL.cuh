#pragma once

#include <md/core/State.cuh>

namespace md {
    class SortedCellList;
    class Top2;
    class Cell;

    class NeighbourList_CLL {
        public:
            NeighbourList_CLL(State& state, float _cutoff, float _margin, SortedCellList& _cll);
            ~NeighbourList_CLL();

            void generate(State& state, Cell* cell);
            void check(State& state, Cell* cell);

            int* get_list() { return this->list; }
            int* get_count() { return this->count; }
            int get_max_neighbours() { return this->max_neighbours; }
            SortedCellList& get_cell_list() { return this->cll; }

            NeighbourList_CLL(const NeighbourList_CLL&) = delete;
            NeighbourList_CLL& operator=(const NeighbourList_CLL&) = delete;
        private:
            SortedCellList& cll;

            float cutoff, margin;
            float Lbox;
            dfloat3 nl_conf; 
            int *list, *count;
            size_t max_neighbours;
            Top2* top2;
            bool* flag;

            // cub用のバッファとそのサイズ
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            // カーネル起動スレッド数
            int generate_nl_num_threads = 0;
            int update_nl_conf_num_threads = 0;
    };
}