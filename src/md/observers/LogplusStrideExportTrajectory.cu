#include <md/observers/LogplusStrideExportTrajectory.cuh>

#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/cells/Cell.cuh>
#include <md/observers/TrajectoryExporter.cuh>

#include <cmath>
#include <string>

using namespace md::observers;

LogplusStrideExportTrajectory::LogplusStrideExportTrajectory(size_t _num_trajectory, float _stride, float _interval, int _counter, bool _is_unwrap, State& state, Cell* _cell, const std::string& output_path)
 : num_trajectory(_num_trajectory), stride(_stride), log_interval(_interval), is_unwrap(_is_unwrap), counters(_num_trajectory), checkers(_num_trajectory), exporters(_num_trajectory) {
    std::fill(counters.data(), counters.data() + num_trajectory, _counter);

    // checkersの初期化（strideずつずらす）
    for (size_t i = 0; i < num_trajectory; i ++) {
        checkers[i] = 1e-3 * std::pow(_interval, _counter) + stride * i;
    }

    // exportersの初期化
    for (size_t i = 0; i < num_trajectory; i ++) {
        std::string path = output_path + "_" + std::to_string(i) + ".xyz";
        exporters[i] = std::make_unique<TrajectoryExporter>(state, path, _cell);
    }
}

void LogplusStrideExportTrajectory::output(State& state) {
    float time = state.dt * state.current_steps;

    for (size_t i = 0; i < num_trajectory; i ++) {
        if (time > checkers[i]) {
            if (is_unwrap) {
                exporters[i]->export_trajectory_unwrap(state);
            }
            else {
                exporters[i]->export_trajectory(state);
            }

            if (i == 0) {
                std::cout << time << ", " << std::flush;
            }

            counters[i] ++;
            checkers[i] = 1e-3 * std::pow(log_interval, counters[i]) + stride * i;
        }
    }
}

void LogplusStrideExportTrajectory::init(State& state) {
    float time = state.dt * state.current_steps;
    std::cout << time << ", " << std::flush;

    for (size_t i = 0; i < num_trajectory; i ++) {
        if (time > checkers[i]) {
            if (is_unwrap) {
                exporters[i]->export_trajectory_unwrap(state);
            }
            else {
                exporters[i]->export_trajectory(state);
            }
        }
    }
}