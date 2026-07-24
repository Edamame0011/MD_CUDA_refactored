#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <memory>

namespace md {
    class State;
    class Interaction;

    namespace observers {
        class EnergiesPrinter {
            public:
                EnergiesPrinter(Interaction* interaction, const std::string& output_path);
                void print_energies(State& state);

            private:
                std::ofstream ofs;
                Interaction* interaction;
        };
    }
}