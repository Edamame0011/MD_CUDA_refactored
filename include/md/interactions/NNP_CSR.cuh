#pragma once

#include <md/interactions/Interaction.cuh>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
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

            torch::inductor::AOTIModelPackageLoader loader;

            NeighbourList* nl = nullptr;
            Cell* cell = nullptr;

            float cutoff;

            int* counts = nullptr;  // それぞれの原子のペア数 (N, )

            // グラフ構造の本体
            int64_t* x_ptr = nullptr;   // (N, )
            int32_t* edge_index_ptr = nullptr;  // (num_edges, )
            int32_t* offsets_ptr = nullptr; // (N + 1, )
            float* edge_weight_ptr = nullptr;

            // torch::Tensor型のラッパー
            std::vector<torch::Tensor> inputs;

            // cub用のバッファ
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
    };
}