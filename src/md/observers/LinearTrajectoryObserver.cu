#include <md/observers/LinearTrajectoryObserver.hpp>
#include <md/observers/TrajectoryExporter.hpp>
#include <md/core/State.hpp>
#include <md/core/Cell.cuh>

namespace md::observers {
    LinearTrajectoryObserver::LinearTrajectoryObserver(int interval, bool is_unwrap, State* state, Cell& cell, Interaction* interaction, const std::string& output_path, const std::vector<std::string>& species_to_symbol)
    : output_interval_(interval), is_unwarp_(is_unwrap) {
        exporter_ = std::make_unique<TrajectoryExporter>(state, cell, interaction, output_path, species_to_symbol);
    }

    void LinearTrajectoryObserver::output(State* state, SimState& simstate) {
        if (simstate.current_steps % output_interval_ == 0) {
            exporter_->export_frame(state, simstate, is_unwarp_);
        }
    }

    void LinearTrajectoryObserver::init(State* state, SimState& simstate) {
        exporter_->export_frame(state, simstate, is_unwarp_);
    }
}