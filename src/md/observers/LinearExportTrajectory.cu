#include <md/observers/LinearExportTrajectory.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <iomanip>
#include <md/cells/CubicCell.cuh>

using namespace md::observers;

template <typename CellType>
LinearExportTrajectory<CellType>::LinearExportTrajectory(int interval, bool _is_unwrap, State& state, const CellType& _cell, const std::string& output_path)
 : output_interval(interval), is_unwrap(_is_unwrap), exporter(state, output_path, _cell) {}

template <typename CellType>
void LinearExportTrajectory<CellType>::output(State& state) {
    if (state.current_steps % this->output_interval == 0) {
        float time = state.dt * state.current_steps;
        std::cout << time << ", ";
        if (is_unwrap) {
            exporter.export_trajectory_unwrap(state);
        } else {
            exporter.export_trajectory(state);
        }
    }
}

template <typename CellType>
void LinearExportTrajectory<CellType>::init(State& state) {
    float time = state.dt * state.current_steps;
    std::cout << time << ", ";
    if (is_unwrap) {
        exporter.export_trajectory_unwrap(state);
    } else {
        exporter.export_trajectory(state);
    }
}

template class LinearExportTrajectory<md::cells::CubicCell>;