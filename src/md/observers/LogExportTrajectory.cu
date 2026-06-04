#include <md/observers/LogExportTrajectory.cuh>

#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/cells/Cell.cuh>

#include <cmath>

using namespace md::observers;

LogExportTrajectory::LogExportTrajectory(float _interval, int _counter, bool _is_unwrap, State& state, Cell* _cell, const std::string& output_path)
 : log_interval(_interval), counter(_counter), is_unwrap(_is_unwrap), exporter(state, output_path, _cell) {
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }

void LogExportTrajectory::output(State& state) {
    // 現在時刻を出力しておく
    float time = state.dt * state.current_steps;
    if (time > checker) {
        std::cout << time << ", " << std::flush;
        if (is_unwrap) {
            exporter.export_trajectory_unwrap(state);
        }
        else {
            exporter.export_trajectory(state);
        }

        this->counter ++;
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }
}

void LogExportTrajectory::init(State& state) {
    float time = state.dt * state.current_steps;
    std::cout << time << ", " << std::flush;
    if (is_unwrap) {
        exporter.export_trajectory_unwrap(state);
    } else {
        exporter.export_trajectory(state);
    }
}