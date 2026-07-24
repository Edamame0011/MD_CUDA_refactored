#pragma once

namespace md {
    class State;
    class Interaction;
    class Integrator;
    class Observer;
    class Cell;

    class Simulator {
        public:
            Simulator(
                State& _state, 
                Interaction *_interaction, 
                Integrator *_integrator, 
                Observer *_observer, 
                Cell* _cell
            ) : state(_state), interaction(_interaction), integrator(_integrator), observer(_observer), cell(_cell) { }
        
            // シミュレーションの実行
            void run(float tsim, int loop_per_graph=100, int log_step=1000);
        
        private:
            State& state;
            Interaction* interaction;
            Integrator* integrator;
            Observer* observer;
            Cell* cell;
    };
}