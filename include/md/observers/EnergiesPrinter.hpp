#pragma once

#include <string>
#include <fstream>
#include <memory>

namespace md {
    class State;
    class SimState;
    class Interaction;

    namespace observers {
        class EnergiesPrinter {
            public:
                EnergiesPrinter(Interaction* interaction, const std::string& output_path);
                void print_energies(State& state, SimState& simstate);

            private:
                float kinetic_energy = 0.0f;
                float potential_energy = 0.0f;
                std::ofstream ofs;
                Interaction* interaction;
        };
    }
}