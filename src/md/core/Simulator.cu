#include <md/core/Simulator.cuh>

#include <md/core/State.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/cells/Cell.cuh>
#include <md/observers/Observer.cuh>

using namespace md;

void Simulator::run(float tsim, int loop_per_graph, int log_step)  {
    if (loop_per_graph > 0) {
        // CUDA Graphsによる最適化のために必要な変数
        cudaGraph_t graph;
        cudaGraphExec_t instance;

        // 録画の開始
        // 複数回のループを一つのグラフとして記録する。
        cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < loop_per_graph; i ++) {
            integrator->integrateStepOne(state);
            cell->apply_pbc(state);
            interaction->calc_force(state);
            integrator->integrateStepTwo(state);
        }
        // 録画の終了
        cudaStreamEndCapture(state.stream, &graph);

        // グラフの変換
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

        // シミュレーション開始時間
        auto start = std::chrono::steady_clock::now();
    
        if (state.current_steps == 0) {
            interaction->calc_force(state);
            observer->init(state);
        }

        int total_steps = static_cast<int>(tsim / state.dt);
        total_steps += state.current_steps;

        int next_print_step = ((state.current_steps / log_step) + 1) * log_step;

        // メインループ
        while (state.current_steps < total_steps) {  
            cudaGraphLaunch(instance, state.stream);
            state.current_steps += loop_per_graph;
            observer->output(state);

            if (state.current_steps >= next_print_step) {
                cudaStreamSynchronize(state.stream);

                auto current_time = std::chrono::steady_clock::now();
                double elapsed_s = std::chrono::duration<double>(current_time - start).count();

                std::cout << state.current_steps << " out of " << total_steps << std::endl;
                std::cout << "経過時間：" << elapsed_s << "s" << std::endl;

                next_print_step = ((state.current_steps / log_step) + 1) * log_step;
            }
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(end - start).count();

        std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;

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

        // シミュレーション開始時間
        auto start = std::chrono::steady_clock::now();

        if (state.current_steps == 0) {
            interaction->calc_force(state);
            observer->init(state);
        }

        int total_steps = static_cast<int>(tsim / state.dt);
        total_steps += state.current_steps;

        int next_print_step = ((state.current_steps / log_step) + 1) * log_step;

        // メインループ
        while (state.current_steps < total_steps) {  
            cudaGraphLaunch(instance, state.stream);
            interaction->calc_force(state);
            integrator->integrateStepTwo(state);
            state.current_steps ++;
            observer->output(state);

            if (state.current_steps >= next_print_step) {
                cudaStreamSynchronize(state.stream);

                auto current_time = std::chrono::steady_clock::now();
                double elapsed_s = std::chrono::duration<double>(current_time - start).count();

                std::cout << "Current steps: " << state.current_steps << " out of " << total_steps << std::endl;
                std::cout << "経過時間：" << elapsed_s << "s" << std::endl;

                next_print_step = ((state.current_steps / log_step) + 1) * log_step;
            }
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(end - start).count();

        std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;

        cudaGraphExecDestroy(instance);
        cudaGraphDestroy(graph);
    }
}