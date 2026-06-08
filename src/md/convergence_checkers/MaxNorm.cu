#include <md/convergence_checkers/MaxNorm.cuh>

#include <md/core/State.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>

namespace {
    struct CalcSquaredNorm {
        dfloat3 force;
        CalcSquaredNorm(dfloat3 f) : force(f) {}
        __device__ float operator() (const size_t idx) const {
            return force.x[idx] * force.x[idx] + force.y[idx] * force.y[idx] + force.z[idx] * force.z[idx];
        }
    };
}

using namespace md::convergence_checkers;

bool MaxNorm::check(State& state) {
    auto N = state.n_atoms;
    float max2 = thrust::transform_reduce(
        thrust::device, 
        thrust::make_counting_iterator<size_t>(0), 
        thrust::make_counting_iterator<size_t>(N), 
        CalcSquaredNorm(state.force), 
        0.0f, 
        thrust::maximum<float>()
    );
    return max2 < (threshold * threshold);
}