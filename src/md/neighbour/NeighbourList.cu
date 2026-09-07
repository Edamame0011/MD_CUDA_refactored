#include <md/neighbour/NeighbourList.hpp>
#include <md/neighbour/NL_utils.cuh>

using Top2 = md::neighbour::Top2;

namespace md {
    NeighbourList::NeighbourList(int n_atoms, int max_neighbours_, float margin_) : max_neighbours(max_neighbours_), margin(margin_) {
        cudaMalloc(&this->nl_conf.x, n_atoms * sizeof(float));
        cudaMalloc(&this->nl_conf.y, n_atoms * sizeof(float));
        cudaMalloc(&this->nl_conf.z, n_atoms * sizeof(float));
        cudaMalloc(&this->top2, sizeof(Top2));

        list.resize(n_atoms * max_neighbours);
        count.resize(n_atoms);
    }

    NeighbourList::~NeighbourList() {
        cudaFree(this->nl_conf.x);
        cudaFree(this->nl_conf.y);
        cudaFree(this->nl_conf.z);
        cudaFree(this->top2);
    }

    void NeighbourList::init_cutoff(const std::vector<float>& cutoff_) {
        const size_t n_pairs = cutoff_.size();
        num_species = (int)std::sqrt(n_pairs);

        std::vector<float> h_cutoff_margin_sq(n_pairs);
        for (int i = 0; i < n_pairs; i ++) {
            const float cutoff_margin = cutoff_[i] + margin;
            h_cutoff_margin_sq[i] = cutoff_margin * cutoff_margin;
        }
        // GPUへ転送
        cutoff_margin_sq = h_cutoff_margin_sq;
    }
}