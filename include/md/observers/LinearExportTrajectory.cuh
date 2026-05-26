#ifndef LINEAR_EXPORT_TRAJECTORY_CUH
#define LINEAR_EXPORT_TRAJECTORY_CUH

#include <md/core/State.cuh>
#include <md/observers/Observer.cuh>

namespace md::observers{
    class LinearExportTrajectory : public Observer {
        public:
            LinearExportTrajectory(int interval) : output_interval(interval) {}
            void output(State& state, Interaction* interaction) override;
            void init(State& state, Interaction* interaction) override;
        private:
            int output_interval;
    };
}

#endif