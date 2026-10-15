#include <md/convergence_checkers/MaxNorm.hpp>

#include <md/core/State.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>

using DeviceVec3 = md::DeviceVec3;

namespace {
    struct CalcSquaredNorm {
        DeviceVec3 force;
        CalcSquaredNorm(DeviceVec3 f) : force(f) {}
        __device__ float operator() (const int idx) const {
            return force.x[idx] * force.x[idx] + force.y[idx] * force.y[idx] + force.z[idx] * force.z[idx];
        }
    };
}

using namespace md::convergence_checkers;

bool MaxNorm::check(State* state, SimState& simstate) {
    auto N = state->n_atoms;
    float max2 = thrust::transform_reduce(
        thrust::cuda::par.on(simstate.stream), 
        thrust::make_counting_iterator<int>(0), 
        thrust::make_counting_iterator<int>(N), 
        CalcSquaredNorm(state->force), 
        0.0f, 
        thrust::maximum<float>()
    );
    return max2 < (threshold * threshold);
}