#ifndef __INTERACTIONS_CUH__
#define __INTERACTIONS_CUH__

#include <md/core/State.cuh>

namespace md {        
    class Interaction {
        public:
            virtual ~Interaction () = default;

            virtual void calc_force(State& state) = 0;
            virtual void calc_potential(State& state) = 0;

        protected:
            Interaction() = default;
    };
}

#endif