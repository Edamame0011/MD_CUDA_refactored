#ifndef __NNP__CUH__
#define __NNP__CUH__

#include <md/core/State.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/utils/NeighbourList.cuh>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>

namespace md::interactions {
    template <typename CellType>
    class NNP : public Interaction {
        public: 
            NNP(State& state, CellType _cell, md::utils::NeighbourList<CellType>* _nl, float _cutoff, int _num_max_edges, const std::string model_path);
            ~NNP();

            void calc_force(State& state) override;
            void calc_potential(State& state) override;

        private: 
            void create_graph(State& state);

            const int num_max_edges;
            int num_edges;

            torch::jit::script::Module model;
            md::utils::NeighbourList<CellType>* nl;
            CellType cell;

            float cutoff;

            int* counts = nullptr;  // それぞれの原子のペア数 (N, )
            int* offsets = nullptr; // それぞれの原子の書き込み位置 (N, )

            // グラフ構造の本体
            int64_t* x_ptr = nullptr;
            float* edge_weight_ptr = nullptr;
            int64_t* edge_index_ptr = nullptr;

            // torch::Tensor型のラッパー
            torch::Tensor x;
            torch::Tensor edge_weight;
            torch::Tensor edge_index;
    };
}

#endif