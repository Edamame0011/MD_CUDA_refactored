#include <md/observers/LogExportTrajectory.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <cmath>
#include <md/cells/CubicCell.cuh>

using namespace md::observers;

template <typename CellType>
LogExportTrajectory<CellType>::LogExportTrajectory(float _interval, int _counter, bool _is_unwrap, State& state, const CellType& _cell, const std::string& output_path)
 : log_interval(_interval), counter(_counter), cell(_cell), exporter(state, output_path) {
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }

template <typename CellType>
void LogExportTrajectory<CellType>::output(State& state) {
    // 現在時刻を出力しておく
    float time = state.dt * state.current_steps;
    if (time > checker) {
        std::cout << time << ", ";
        if (is_unwrap) {
            exporter.export_trajectory_unwrap<CellType>(state, cell);
        }
        else {
            exporter.export_trajectory(state);
        }

        this->counter ++;
        this->checker = 1e-3 * std::pow(log_interval, counter);
    }
}
template <typename CellType>
void LogExportTrajectory<CellType>::init(State& state) {
    float time = state.dt * state.current_steps;
    std::cout << time << ", ";
    if (is_unwrap) {
        exporter.export_trajectory_unwrap<CellType>(state, cell);
    } else {
        exporter.export_trajectory(state);
    }
}

template class LogExportTrajectory<md::cells::CubicCell>;