#pragma once

namespace md {
    class State;
    class Interaction;
    class Integrator;
    class Observer;
    class Cell;
}

namespace md {
    class Simulator {
        public:
            // コンストラクタ
            Simulator(
                State& _state, 
                Interaction *_interaction, 
                Integrator *_integrator, 
                Observer *_observer, 
                Cell* _cell
            ) : state(_state), interaction(_interaction), integrator(_integrator), observer(_observer), cell(_cell) { }
        
            // シミュレーションの実行
            void run(float tsim, bool use_cuda_graphs=true);
        
        private:
            State& state;
            Interaction* interaction;
            Integrator* integrator;
            Observer* observer;
            Cell* cell;
    };
}