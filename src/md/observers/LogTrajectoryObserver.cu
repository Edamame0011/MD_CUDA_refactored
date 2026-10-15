#include <md/observers/LogTrajectoryObserver.hpp>
#include <md/observers/TrajectoryExporter.hpp>
#include <md/core/State.hpp>
#include <md/core/Cell.cuh>

#include <cmath>

namespace md::observers {
    LogTrajectoryObserver::LogTrajectoryObserver(float interval, int counter, bool is_unwrap, State* state, Cell& cell, Interaction* interaction, const std::string& output_path, const std::string& temp_path, const std::vector<std::string>& species_to_symbol)
    : log_interval_(interval), counter_(counter), is_unwarp_(is_unwrap) {
        exporter_ = std::make_unique<TrajectoryExporter>(state, cell, interaction, output_path, species_to_symbol);
        ofs_.open(temp_path);
    }

    void LogTrajectoryObserver::output(State* state, SimState& simstate) {
        float time = simstate.dt * simstate.current_steps;
        if (time > checker_) {
            ofs_ << time << "," << std::flush;
            exporter_->export_frame(state, simstate, is_unwarp_);
            counter_ ++;
            checker_ = 1e-3 * std::pow(log_interval_, counter_);
        }
    }

    void LogTrajectoryObserver::init(State* state, SimState& simstate) {
        float time = simstate.dt * simstate.current_steps;
        ofs_ << time << ", " << std::flush;
        exporter_->export_frame(state, simstate, is_unwarp_);
    }
}