#pragma once

namespace md {
    class State;
    class SimState;
    class Interaction;
    class Integrator;
    class Observer;
    class Cell;

    class Simulator {
        public:
            Simulator(
                State& state_, 
                SimState& simstate_, 
                Interaction *interaction_, 
                Integrator *integrator_, 
                Observer *observer_, 
                Cell& cell_
            ) : state(state_), simstate(simstate_), interaction(interaction_), integrator(integrator_), observer(observer_), cell(cell_) { }
        
            // シミュレーションの実行
            void run(float tsim, int loop_per_graph=100, int log_step=1000);
        
        private:
            State& state;
            SimState& simstate;
            Cell& cell;
            Interaction* interaction;
            Integrator* integrator;
            Observer* observer;
    };
}