#pragma once

#include <md/observers/Observer.cuh>
#include <memory>
#include <vector>
namespace md {
    class Cell;
    
    namespace observers {
        class TrajectoryExporter;
    }
}

namespace md::observers{
    class LogplusStrideExportTrajectory : public Observer {
        public:
            LogplusStrideExportTrajectory(size_t num_trajectory, float stride, float _interval, int _counter, bool _is_unwrap, State& state, Cell* _cell, const std::string& output_path);
            void output(State& state) override;
            void init(State& state) override;
        private:
            size_t num_trajectory;
            float stride;
            float log_interval;
            bool is_unwrap;
            std::vector<int> counters;
            std::vector<float> checkers;
            std::vector<std::unique_ptr<TrajectoryExporter>> exporters;
    };
}