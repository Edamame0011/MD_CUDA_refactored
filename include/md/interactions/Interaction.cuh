#ifndef INTERACTIONS_CUH
#define INTERACTIONS_CUH

#include <md/core/State.cuh>

namespace md {        
    class Interaction {
        public:
            virtual ~Interaction () = default;

            virtual void forward(State& state) = 0;

        protected:
            Interaction() = default;
    };
}

#endif