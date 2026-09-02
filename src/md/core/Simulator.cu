#include <md/core/Simulator.hpp>

#include <md/core/State.hpp>
#include <md/integrators/Integrator.hpp>
#include <md/interactions/Interaction.hpp>
#include <md/core/Cell.cuh>
#include <md/observers/Observer.hpp>

#include <chrono>
#include <iostream>

using namespace md;

void Simulator::run(float tsim, int loop_per_graph, int log_step)  {
    if (loop_per_graph > 0) {
        // CUDA Graphsによる最適化のために必要な変数
        cudaGraph_t graph;
        cudaGraphExec_t instance;

        // 録画の開始
        // 複数回のループを一つのグラフとして記録する。
        cudaStreamBeginCapture(simstate.stream, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < loop_per_graph; i ++) {
            integrator->integrateStepOne(state, simstate);
            cell.apply_pbc(state, simstate);
            interaction->calc_force(state, simstate);
            integrator->integrateStepTwo(state, simstate);
        }
        // 録画の終了
        cudaStreamEndCapture(simstate.stream, &graph);

        // グラフの変換
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

        // シミュレーション開始時間
        auto start = std::chrono::steady_clock::now();
    
        if (simstate.current_steps == 0) {
            interaction->calc_force(state, simstate);
            observer->init(state);
        }

        int total_steps = static_cast<int>(tsim / simstate.dt);
        total_steps += simstate.current_steps;

        int next_print_step = ((simstate.current_steps / log_step) + 1) * log_step;

        // メインループ
        while (simstate.current_steps < total_steps) {  
            cudaGraphLaunch(instance, simstate.stream);
            simstate.current_steps += loop_per_graph;
            observer->output(state, simstate);

            if (simstate.current_steps >= next_print_step) {
                cudaStreamSynchronize(simstate.stream);

                auto current_time = std::chrono::steady_clock::now();
                double elapsed_s = std::chrono::duration<double>(current_time - start).count();

                std::cout << simstate.current_steps << " out of " << total_steps << std::endl;
                std::cout << "経過時間：" << elapsed_s << "s" << std::endl;

                next_print_step = ((simstate.current_steps / log_step) + 1) * log_step;
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
        cudaStreamBeginCapture(simstate.stream, cudaStreamCaptureModeGlobal);
        integrator->integrateStepOne(state, simstate);
        cell.apply_pbc(state, simstate);
        cudaStreamEndCapture(simstate.stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

        // シミュレーション開始時間
        auto start = std::chrono::steady_clock::now();

        if (simstate.current_steps == 0) {
            interaction->calc_force(state, simstate);
            observer->init(state);
        }

        int total_steps = static_cast<int>(tsim / simstate.dt);
        total_steps += simstate.current_steps;

        int next_print_step = ((simstate.current_steps / log_step) + 1) * log_step;

        // メインループ
        while (simstate.current_steps < total_steps) {  
            cudaGraphLaunch(instance, simstate.stream);
            interaction->calc_force(state, simstate);
            integrator->integrateStepTwo(state, simstate);
            simstate.current_steps ++;
            observer->output(state, simstate);

            if (simstate.current_steps >= next_print_step) {
                cudaStreamSynchronize(simstate.stream);

                auto current_time = std::chrono::steady_clock::now();
                double elapsed_s = std::chrono::duration<double>(current_time - start).count();

                std::cout << "Current steps: " << simstate.current_steps << " out of " << total_steps << std::endl;
                std::cout << "経過時間：" << elapsed_s << "s" << std::endl;

                next_print_step = ((simstate.current_steps / log_step) + 1) * log_step;
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