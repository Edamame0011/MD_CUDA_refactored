#ifndef OBSERVER_CUH
#define OBSERVER_CUH

#include <md/core/State.cuh>

namespace md{
    class Observer {
        public:
            virtual ~Observer () = default;

            virtual void output(const State& state) = 0;
        protected:
            Observer() = default;
    };
}

#endif

namespace md::observers {
    void print_energies(const State& state);
}