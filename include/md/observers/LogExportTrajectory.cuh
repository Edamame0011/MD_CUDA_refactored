#pragma once

#include <md/observers/Observer.cuh>
#include <memory>
#include <fstream>

namespace md {
    class Cell;
    
    namespace observers {
        class TrajectoryExporter;
    }
}

namespace md::observers{
    class LogExportTrajectory : public Observer {
        public:
            LogExportTrajectory(float _interval, int _counter, bool _is_unwrap, State& state, Cell* _cell, const std::string& output_path, const std::string& temp_path);
            void output(State& state) override;
            void init(State& state) override;
        private:
            float log_interval;
            int counter;
            float checker;
            bool is_unwrap;
            std::unique_ptr<TrajectoryExporter> exporter;
            std::ofstream ofs;
    };
}