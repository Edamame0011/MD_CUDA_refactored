#ifndef __LINEAR_EXPORT_TRAJECTORY_CUH__
#define __LINEAR_EXPORT_TRAJECTORY_CUH__

#include <md/core/State.cuh>
#include <md/observers/Observer.cuh>
#include <md/observers/TrajectoryExporter.cuh>

namespace md::observers{
    template <typename CellType>
    class LinearExportTrajectory : public Observer {
        public:
            LinearExportTrajectory(int interval, bool _is_unwrap, State& state, const CellType& _cell, const std::string& output_path);
            void output(State& state) override;
            void init(State& state) override;
        private:
            int output_interval;
            bool is_unwrap;
            TrajectoryExporter<CellType> exporter;
    };
}

#endif