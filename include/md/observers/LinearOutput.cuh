#ifndef __LINEAR_OUTPUT_CUH__
#define __LINEAR_OUTPUT_CUH__

#include <md/core/State.cuh>
#include <md/observers/Observer.cuh>
#include <md/interactions/Interaction.cuh>

namespace md::observers{
    class LinearOutput : public Observer {
        public:
            LinearOutput(int interval, Interaction* _interaction) : output_interval(interval), interaction(_interaction) {}
            void output(State& state) override;
            void init(State& state) override;
        private:
            int output_interval;
            Interaction* interaction;
    };
}

#endif