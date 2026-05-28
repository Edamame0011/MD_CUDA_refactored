#include <md/interactions/NNP.cuh>
#include <md/cells/CubicCell.cuh>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
// #include <torch_tensorrt/torch_tensorrt.h>

namespace {
    template <typename CellType>
    __global__ void count_pairs_kernel(
        dfloat3 pos, 
        int* __restrict__ counts, 
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        CellType cell, 
        const float cutoff, 
        const int num_atoms, 
        const int max_neighbours
    ) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_atoms) return;

        const float pxi = pos.x[idx];
        const float pyi = pos.y[idx];
        const float pzi = pos.z[idx];

        int valid_pairs = 0;
        for (int c = 0; c < count[idx]; c ++) {
            int j = list[idx * max_neighbours + c];

            if (idx >= j) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];
            
            float dx = pxi - pxj;
            float dy = pyi - pyj;
            float dz = pzi - pzj;
        
            cell.apply_pbc(dx, dy, dz);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            
            if (dist_sq < cutoff * cutoff) {
                valid_pairs++;
            }
        }
        counts[idx] = valid_pairs;
    }

    template <typename CellType>
    __global__ void build_graph_kernel(
        dfloat3 pos, 
        int64_t* __restrict__ edge_index_ptr, 
        float* __restrict__ edge_weight_ptr, 
        const int* __restrict__ offsets,
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        CellType cell, 
        const float cutoff, 
        const int num_atoms, 
        const int max_neighbours, 
        const int num_edges,
        const int num_pairs
    ) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_atoms) return;

        const float pxi = pos.x[idx];
        const float pyi = pos.y[idx];
        const float pzi = pos.z[idx];

        int write_idx = offsets[idx];

        for (int c = 0; c < count[idx]; c ++) {
            int j = list[idx * max_neighbours + c];
            if (idx >= j) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];

            float dx = pxj - pxi;
            float dy = pyj - pyi;
            float dz = pzj - pzi;
        
            cell.apply_pbc(dx, dy, dz);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            
            if (dist_sq < cutoff * cutoff) {
                // i -> j
                edge_index_ptr[write_idx] = idx;
                edge_index_ptr[num_edges + write_idx] = j;

                edge_weight_ptr[write_idx] = dx;
                edge_weight_ptr[num_edges + write_idx] = dy;
                edge_weight_ptr[2 * num_edges + write_idx] = dz;

                // j -> iへコピー
                int rev_idx = write_idx + num_pairs;
                edge_index_ptr[rev_idx] = j;
                edge_index_ptr[num_edges + rev_idx] = idx;

                edge_weight_ptr[rev_idx] = -dx;
                edge_weight_ptr[num_edges + rev_idx] = -dy;
                edge_weight_ptr[2 * num_edges + rev_idx] = -dz;

                write_idx ++;
            }
        }
    }
}

using namespace md::interactions;

template <typename CellType>
NNP<CellType>::NNP(
    State& state, 
    CellType _cell, 
    md::utils::NeighbourList<CellType>* _nl, 
    float _cutoff, 
    int _num_max_edges, 
    const std::string model_path
) : cell(_cell), cutoff(_cutoff), nl(_nl), num_max_edges(_num_max_edges) {
    // モデルの読み込み
    try {
        model = torch::jit::load(model_path, torch::kCUDA);
        std::cout << "モデルを読み込みました：" << model_path << std::endl;
    }
    catch(c10::Error& e) {
        std::cerr << "モデルの読み込みに失敗しました。" << std::endl
                  << e.what() << std::endl;
        throw;
    }
    model.eval();

    auto N = state.n_atoms;
    auto view = state.get_view();

    // メモリの確保
    cudaMalloc(&x_ptr, N * sizeof(int64_t));
    cudaMalloc(&edge_weight_ptr, 3 * num_max_edges * sizeof(float));
    cudaMalloc(&edge_index_ptr, 2 * num_max_edges * sizeof(int64_t));
    cudaMalloc(&counts, N * sizeof(int));
    cudaMalloc(&offsets, N * sizeof(int));

    // 原子番号はシミュレーションを通して変わらないため、最初に初期化する
    // int32_t -> int64_t
    thrust::copy(
        thrust::device, 
        view.atomic_numbers, 
        view.atomic_numbers + N, 
        x_ptr
    );
}

template <typename CellType>
NNP<CellType>::~NNP() {
    cudaFree(x_ptr);
    cudaFree(edge_weight_ptr);
    cudaFree(edge_index_ptr);
    cudaFree(counts);
    cudaFree(offsets);
}

template <typename CellType>
void NNP<CellType>::create_graph(State& state) {
    int N = state.n_atoms;
    auto view = state.get_view();

    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;

    count_pairs_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        view.pos, 
        counts, 
        nl->get_list(), 
        nl->get_count(), 
        cell, 
        cutoff, 
        N, 
        nl->get_max_neighbours()
    );

    // 手前のインデックスまでを加算
    thrust::exclusive_scan(
        thrust::cuda::par.on(state.stream),
        counts,
        counts + N,
        offsets
    );

    int last_count, last_offset;
    cudaMemcpyAsync(&last_count, counts + N - 1, sizeof(int), cudaMemcpyDeviceToHost, state.stream);
    cudaMemcpyAsync(&last_offset, offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost, state.stream);

    cudaStreamSynchronize(state.stream);

    int num_pairs = last_offset + last_count;
    num_edges = 2 * num_pairs;

    build_graph_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        view.pos, 
        edge_index_ptr, 
        edge_weight_ptr, 
        offsets, 
        nl->get_list(), 
        nl->get_count(), 
        cell, 
        cutoff, 
        N, 
        nl->get_max_neighbours(), 
        num_edges, 
        num_pairs
    );
}

template <typename CellType>
void NNP<CellType>::calc_force(State& state) {
    int N = state.n_atoms;
    nl->check(state, cell);
    create_graph(state);

    auto view = state.get_view();

    auto opt = torch::TensorOptions().device(torch::kCUDA);

    x = torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64));
    edge_index = torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt64));
    edge_weight = torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32)).set_requires_grad(true);

    // ストリームを指定
    c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromExternal(state.stream, x.device().index());
    c10::cuda::CUDAStreamGuard guard(torch_stream);

    auto result_iv = model.forward({x, edge_index, edge_weight});

    auto result_tuple = result_iv.toTuple();
    auto elements = result_tuple->elements();

    auto energy = elements[0].toTensor().to(torch::kFloat32).detach();
    auto forces = elements[1].toTensor().to(torch::kFloat32).detach();

    // libtorch側のポインター
    float* forces_ptr = forces.data_ptr<float>();

    // 値のコピー
    cudaMemcpyAsync(view.force.x, forces_ptr, N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream);
    cudaMemcpyAsync(view.force.y, forces_ptr + N, N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream);
    cudaMemcpyAsync(view.force.z, forces_ptr + 2 * N, N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream);
}

template <typename CellType>
void NNP<CellType>::calc_potential(State& state) {
    nl->check(state, cell);
    create_graph(state);

    int N = state.n_atoms;

    auto view = state.get_view();

    auto opt = torch::TensorOptions().device(torch::kCUDA);

    x = torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64));
    edge_index = torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt64));
    edge_weight = torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32)).set_requires_grad(true);

    c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromExternal(state.stream, x.device().index());
    c10::cuda::CUDAStreamGuard guard(torch_stream);

    auto result_iv = model.forward({x, edge_index, edge_weight});

    auto result_tuple = result_iv.toTuple();
    auto elements = result_tuple->elements();

    auto energy = elements[0].toTensor().to(torch::kFloat32).detach();
    auto forces = elements[1].toTensor().to(torch::kFloat32).detach();

    float* energy_ptr = energy.data_ptr<float>();

    cudaMemcpy(&state.potential_energy, energy_ptr, sizeof(float), cudaMemcpyDeviceToHost);
}

template class NNP<md::cells::CubicCell>;