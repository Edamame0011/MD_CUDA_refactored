#pragma once

#include <md/observers/Observer.cuh>
#include <memory>

namespace md::observers{
    class EnergiesPrinter;

    class LinearOutput : public Observer {
        public:
            LinearOutput(int interval, Interaction* _interaction, const std::string& output_path);
            ~LinearOutput();
            void output(State& state) override;
            void init(State& state) override;
        private:
            int output_interval;
            std::unique_ptr<EnergiesPrinter> printer;
    };
}