#include <md/core/Simulator.cuh>

#include <md/core/State.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/cells/Cell.cuh>
#include <md/observers/Observer.cuh>

using namespace md;

void Simulator::run(float tsim, bool use_cuda_graphs)  {
    if (use_cuda_graphs) {
        // CUDA Graphsによる最適化のために必要な変数
        cudaGraph_t graph;
        cudaGraphExec_t instance;

        // 録画の開始
        // 複数回のループを一つのグラフとして記録する。
        constexpr int num_loop_per_graph = 100;
        cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < num_loop_per_graph; i ++) {
            integrator->integrateStepOne(state);
            cell->apply_pbc(state);
            interaction->calc_force(state);
            integrator->integrateStepTwo(state);
        }
        // 録画の終了
        cudaStreamEndCapture(state.stream, &graph);

        // グラフの変換
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

        if (state.current_steps == 0) {
            interaction->calc_force(state);
            observer->init(state);
        }

        int total_steps = static_cast<int>(tsim / state.dt);
        total_steps += state.current_steps;

        // メインループ
        while (state.current_steps < total_steps) {  
            cudaGraphLaunch(instance, state.stream);
            state.current_steps += num_loop_per_graph;
            observer->output(state);
        }

        cudaGraphExecDestroy(instance);
        cudaGraphDestroy(graph);
    } else {
        // 前半のみをグラフに記録
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeGlobal);
        integrator->integrateStepOne(state);
        cell->apply_pbc(state);
        cudaStreamEndCapture(state.stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

        if (state.current_steps == 0) {
            interaction->calc_force(state);
            observer->init(state);
        }

        int total_steps = static_cast<int>(tsim / state.dt);
        total_steps += state.current_steps;

        // メインループ
        while (state.current_steps < total_steps) {  
            cudaGraphLaunch(instance, state.stream);
            interaction->calc_force(state);
            integrator->integrateStepTwo(state);
            state.current_steps ++;
            observer->output(state);
        }

        cudaGraphExecDestroy(instance);
        cudaGraphDestroy(graph);
    }
}