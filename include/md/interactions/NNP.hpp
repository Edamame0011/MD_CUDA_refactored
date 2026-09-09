#pragma once

#include <md/interactions/Interaction.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace md {
    class Cell;
    class NeighbourList;

    namespace interactions {
        class NNP : public Interaction {
            public: 
                NNP(State* state, Cell& cell, NeighbourList* nl, float cutoff, int num_max_edges, const std::string model_path);
                ~NNP();

                void calc_force(State* state, SimState& simstate) override;
                float calc_potential(State* state, SimState& simstate) override;

            private: 
                void create_graph(State* state, SimState& simstate);

                const int num_max_edges;
                int num_edges;

                torch::inductor::AOTIModelPackageLoader loader;

                NeighbourList* nl = nullptr;
                Cell& cell;

                float cutoff;

                int* counts = nullptr;  // それぞれの原子のペア数 (N, )
                int* offsets = nullptr; // それぞれの原子の書き込み位置 (N, ) 

                // グラフ構造の本体
                int64_t* x_ptr = nullptr;
                float* edge_weight_ptr = nullptr;
                int64_t* edge_index_ptr = nullptr;

                // torch::Tensor型のラッパー
                std::vector<torch::Tensor> inputs;
        };
    }
}