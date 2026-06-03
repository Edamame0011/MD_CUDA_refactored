#ifndef __LINEAR_EXPORT_TRAJECTORY_CUH__
#define __LINEAR_EXPORT_TRAJECTORY_CUH__

#include <md/core/State.cuh>
#include <md/observers/Observer.cuh>
#include <md/observers/TrajectoryExporter.cuh>
#include <md/cells/Cell.cuh>

namespace md::observers{
    class LinearExportTrajectory : public Observer {
        public:
            LinearExportTrajectory(int interval, bool _is_unwrap, State& state, Cell* _cell, const std::string& output_path);
            void output(State& state) override;
            void init(State& state) override;
        private:
            int output_interval;
            bool is_unwrap;
            TrajectoryExporter exporter;
    };
}

#endif