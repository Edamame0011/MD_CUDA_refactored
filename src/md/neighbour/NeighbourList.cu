#include <md/neighbour/NeighbourList.hpp>
#include <md/neighbour/NL_utils.cuh>

using Top2 = md::neighbour::Top2;

namespace md {
    NeighbourList::NeighbourList(int n_atoms, int max_neighbours_) : max_neighbours(max_neighbours_) {
        cudaMalloc(&this->nl_conf.x, n_atoms * sizeof(float));
        cudaMalloc(&this->nl_conf.y, n_atoms * sizeof(float));
        cudaMalloc(&this->nl_conf.z, n_atoms * sizeof(float));
        cudaMalloc(&this->top2, sizeof(Top2));
        cudaMalloc(&this->flag, sizeof(bool));
        cudaMemset(this->flag, 1, sizeof(bool));

        list.resize(n_atoms * max_neighbours);
        count.resize(n_atoms);
    }

    NeighbourList::~NeighbourList() {
        cudaFree(this->nl_conf.x);
        cudaFree(this->nl_conf.y);
        cudaFree(this->nl_conf.z);
        cudaFree(this->top2);
        cudaFree(this->flag);
    }
}