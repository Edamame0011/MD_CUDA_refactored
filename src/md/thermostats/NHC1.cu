#include <md/thermostats/NHC1.cuh>
#include <md/core/State.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <thrust/iterator/counting_iterator.h>
#include <cmath>

namespace {
    __global__ void calc_scaling_factor (
        float *kinetic_energy, 
        ChainState c_state, 
        float m0_inv, 
        float targ_kin, 
        float dt
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            auto AKIN = 2.0f * *kinetic_energy;
            auto dt_half = 0.5f * dt;
            auto dt_quarter = 0.25f * dt;

            auto f0 = *c_state.force;
            auto v0 = *c_state.vel;
            auto p0 = *c_state.pos;

            // 逆順の更新
            f0 = (AKIN - targ_kin) * m0_inv;
            v0 += dt_quarter * f0;

            // スケーリング
            float sf = exp(-dt_half * v0);
            AKIN *= exp(-dt * v0);

            // 変位の更新
            p0 += dt_half * v0;

            // 順方向の更新
            f0 = (AKIN - targ_kin) * m0_inv;
            v0 += dt_quarter * f0;

            // グローバルメモリに書き込み
            *c_state.force = f0;
            *c_state.vel = v0;
            *c_state.pos = p0;
            *c_state.scaling_factor = sf;
        }
    }

    struct Scaling {
        dfloat3 vel;
        float* scaling_factor;
        Scaling(dfloat3 _vel, float* _sf) : vel(_vel), scaling_factor(_sf) {}
        __device__ void operator() (int idx) {
            auto sf = *scaling_factor;
            vel.x[idx] *= sf;
            vel.y[idx] *= sf;
            vel.z[idx] *= sf;
        }
    };
}

using namespace md::thermostats;

NHC1::NHC1(const float _tau, TemperatureScheduler *_scheduler)
 :  tau(_tau), scheduler(_scheduler) {
    cudaMalloc(&c_state.pos, sizeof(float));
    cudaMalloc(&c_state.vel, sizeof(float));
    cudaMalloc(&c_state.force, sizeof(float));
    cudaMalloc(&c_state.scaling_factor, sizeof(float));
}

NHC1::~NHC1() {
    cudaFree(c_state.pos);
    cudaFree(c_state.vel);
    cudaFree(c_state.force);
    cudaFree(c_state.scaling_factor);
}

void NHC1::init(State& state) {
    this->dof = 3 * state.n_atoms;
    this->calculator = std::make_unique<KinEnergyCalculator>(state);

    float zero = 0.0f;
    cudaMemcpy(c_state.pos, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.vel, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.force, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_state.scaling_factor, &zero, sizeof(float), cudaMemcpyHostToDevice);
}

void NHC1::stepOne(State& state) {
    op(state);
}

void NHC1::stepTwo(State& state) {
    op(state);
}

void NHC1::op(State& state) {
    auto N = state.n_atoms;
    auto view = state.get_view();
    auto target_temperature = this->scheduler->get_temperature(state.current_steps);
    c_mass = tau * tau * boltzmann_constant * target_temperature * dof;

    calculator->calc_kinetic_energy(state);
    
    calc_scaling_factor<<<1, 1, 0, state.stream>>>(
        view.kinetic_energy, 
        this->c_state, 
        1.0f / c_mass, 
        target_temperature * dof * boltzmann_constant, 
        state.dt
    );

    thrust::for_each(
        thrust::cuda::par.on(state.stream),  
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        Scaling(
            view.vel, 
            c_state.scaling_factor
        )
    );
}