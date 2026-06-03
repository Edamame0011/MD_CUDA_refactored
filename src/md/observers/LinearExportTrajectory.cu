#include <md/observers/LinearExportTrajectory.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <iomanip>
#include <md/cells/CubicCell.cuh>

using namespace md::observers;

LinearExportTrajectory::LinearExportTrajectory(int interval, bool _is_unwrap, State& state, Cell* _cell, const std::string& output_path)
 : output_interval(interval), is_unwrap(_is_unwrap), exporter(state, output_path, _cell) {}

void LinearExportTrajectory::output(State& state) {
    if (state.current_steps % this->output_interval == 0) {
        float time = state.dt * state.current_steps;
        std::cout << time << ", " << std::flush;
        if (is_unwrap) {
            exporter.export_trajectory_unwrap(state);
        } else {
            exporter.export_trajectory(state);
        }
    }
}

void LinearExportTrajectory::init(State& state) {
    float time = state.dt * state.current_steps;
    std::cout << time << ", " << std::flush;
    if (is_unwrap) {
        exporter.export_trajectory_unwrap(state);
    } else {
        exporter.export_trajectory(state);
    }
}