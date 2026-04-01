#include <md/thermostats/BussiThermostat.cuh>
#include <md/core/constant.h>
#include <thrust/transform_reduce.h>
#include <md/utils/compute.cuh>

namespace {
    __global__ void calc_scaling_factor(
        float *kinetic_energy, 
        float *scaling_factor, 
        float r1, 
        float r2, 
        float g, 
        float dof, 
        float targ_kin, 
        float tau_inv, 
        float dt  
    ) {
        float f = expf(-dt * tau_inv);
        float alpha2 = f + (targ_kin * (1 - f) * (r1 * r1 + r2 * r2 + 2 * g)) / (dof * *kinetic_energy) + 2 * r1 * sqrtf((targ_kin * f * (1 - f)) / (dof * *kinetic_energy));

        *scaling_factor = sqrtf(alpha2);
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

void BussiThermostat::init(const State& state) {
    this->dof = 3 * state.n_atoms;
    this->gamma_dist = std::gamma_distribution<float>((dof - 2) / 2.0f, 1.0f);
}

void BussiThermostat::stepTwo(State& state) {
    float target_temperature = this->scheduler->get_temperature(state.current_steps);
    float targ_kin = (dof * boltzmann_constant * target_temperature) / 2;
    
    calculator->calc_kinetic_energy(state);

    // 乱数を生成
    float r1 = this->normal_dist(this->gen);
    float r2 = this->normal_dist(this->gen);
    float g = this->gamma_dist(this->gen);

    auto view = state.get_view();

    // スケーリング要素を計算
    calc_scaling_factor<<<1, 1>>>(
        view.kinetic_energy, 
        this->scaling_factor, 
        r1, 
        r2, 
        g, 
        this->dof, 
        targ_kin, 
        1.0f / tau, 
        state.dt
    );

    // スケーリング
    thrust::for_each(
        thrust::device, 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(state.n_atoms), 
        Scaling(
            view.vel, 
            this->scaling_factor
        )
    ); 
}