#include <md/observers/LinearOutput.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <iomanip>

using namespace md::observers;

void LinearOutput::output(const State& state, const int step) {
    if (step % this->output_interval == 0) {
        print_energies(state, step);
    }
}