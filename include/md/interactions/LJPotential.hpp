#pragma once

#include <md/interactions/Interaction.hpp>
#include <thrust/device_vector.h>
#include <vector>

namespace md {
    class Cell;
    class NeighbourList;

    namespace interactions {
        struct lj_params {
            thrust::device_vector<float> sigma, sigma6, epsilon, cutoff, cutoff_sq, deriv_1st_LJpotential_cutoff;
        };

        class LJPotential: public Interaction {
            public: 
                LJPotential(
                    int _num_species, 
                    Cell& cell, 
                    NeighbourList *NL, 
                    std::vector<float> sigma, 
                    std::vector<float> epsilon, 
                    std::vector<float> cutoff
                );

                void calc_force(State& state, SimState& simstate) override;
                float calc_potential(State& state, SimState& simstate) override;

            private: 
                int num_species;
                lj_params params;

                Cell& cell;
                NeighbourList *nl;
        };
    }
}