#ifndef __LOG_EXPORT_TRAJECTORY_CUH__
#define __LOG_EXPORT_TRAJECTORY_CUH__

#include <md/core/State.cuh>
#include <md/observers/Observer.cuh>
#include <md/observers/TrajectoryExporter.cuh>
#include <md/cells/Cell.cuh>

namespace md::observers{
    class LogExportTrajectory : public Observer {
        public:
            LogExportTrajectory(float _interval, int _counter, bool _is_unwrap, State& state, Cell* _cell, const std::string& output_path);
            void output(State& state) override;
            void init(State& state) override;
        private:
            float log_interval;
            int counter;
            float checker;
            bool is_unwrap;
            TrajectoryExporter exporter;
    };
}

#endif