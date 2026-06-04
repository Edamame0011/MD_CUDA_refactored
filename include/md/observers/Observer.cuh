#pragma once

namespace md{
    class State;
    class Interaction;

    class Observer {
        public:
            virtual ~Observer () = default;

            virtual void output(State& state) = 0;
            virtual void init(State& state) = 0;
        protected:
            Observer() = default;
    };
}

namespace md::observers {
    void print_energies(State& state, Interaction* interaction);
}