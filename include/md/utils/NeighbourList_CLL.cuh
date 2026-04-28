#ifndef __NEIGHBOUR_LIST_CLL_CUH__
#define __NEIGHBOUR_LIST_CLL_CUH__

#include <md/utils/CellList.cuh>
#include <md/cells/CubicCell.cuh>
#include <md/utils/NeighbourList.cuh>

namespace md::utils {
    class NeighbourList_CLL {
        public:
            NeighbourList_CLL(State& state, float _cutoff, float _margin, CellList& _cll);
            ~NeighbourList_CLL();

            void generate(State& state, md::cells::CubicCell cell);
            void check(State& state, md::cells::CubicCell cell);

            unsigned int* get_list() { return this->list; }
            unsigned int* get_count() { return this->count; }
            int get_max_neighbours() { return this->max_neighbours; }
            CellList& get_cell_list() { return this->cll; }

            NeighbourList_CLL(const NeighbourList_CLL&) = delete;
            NeighbourList_CLL& operator=(const NeighbourList_CLL&) = delete;
        private:
            CellList& cll;

            float cutoff, margin;
            float Lbox;
            dfloat3 nl_conf; 
            unsigned int *list, *count;
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

#endif