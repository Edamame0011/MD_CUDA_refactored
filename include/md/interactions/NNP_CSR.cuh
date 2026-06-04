#pragma once

#include <md/interactions/Interaction.cuh>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>

namespace md {
    class State;
    class Cell;
    class NeighbourList;
}

namespace md::interactions {
    class NNP_CSR : public Interaction {
        public: 
            NNP_CSR(State& state, Cell* _cell, NeighbourList* _nl, float _cutoff, int _num_max_edges, const std::string model_path);
            ~NNP_CSR();

            void calc_force(State& state) override;
            void calc_potential(State& state) override;

        private: 
            void create_graph(State& state);

            const int num_max_edges;
            int num_edges;

            torch::jit::script::Module model;
            NeighbourList* nl;
            Cell* cell;

            float cutoff;

            int* counts = nullptr;  // それぞれの原子のペア数 (N, )

            // グラフ構造の本体
            int64_t* x_ptr = nullptr;   // (N, )
            int64_t* edge_index_ptr = nullptr;  // (num_edges, )
            int64_t* offsets_ptr = nullptr; // (N + 1, )
            float* edge_weight_ptr = nullptr;

            // torch::Tensor型のラッパー
            torch::Tensor x;
            torch::Tensor edge_index;
            torch::Tensor edge_weight;
            torch::Tensor offsets;

            // cub用のバッファ
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
    };
}