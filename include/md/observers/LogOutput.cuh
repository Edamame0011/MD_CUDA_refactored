#pragma once

#include <md/observers/Observer.cuh>
#include <memory>
#include <string>

namespace md::observers{
    class EnergiesPrinter;

    class LogOutput : public Observer {
        public:
            LogOutput(float _interval, int _counter, Interaction* _interaction, const std::string& output_path);
            ~LogOutput();
            void output(State& state) override;
            void init(State& state) override;
        private:
            float log_interval;
            int counter;
            float checker;
            std::unique_ptr<EnergiesPrinter> printer;
    };
}