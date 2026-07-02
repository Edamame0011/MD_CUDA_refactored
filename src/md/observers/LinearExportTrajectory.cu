#include <md/observers/LinearExportTrajectory.cuh>

#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/cells/Cell.cuh>

#include <iomanip>

using namespace md::observers;

LinearExportTrajectory::LinearExportTrajectory(int interval, bool _is_unwrap, State& state, Cell* _cell, const std::string& output_path)
 : output_interval(interval), is_unwrap(_is_unwrap), exporter(state, output_path, _cell) {}

void LinearExportTrajectory::output(State& state) {
    if (state.current_steps % this->output_interval == 0) {
        if (is_unwrap) {
            exporter.export_trajectory_unwrap(state);
        } else {
            exporter.export_trajectory(state);
        }
    }
}

void LinearExportTrajectory::init(State& state) {
    if (is_unwrap) {
        exporter.export_trajectory_unwrap(state);
    } else {
        exporter.export_trajectory(state);
    }
}