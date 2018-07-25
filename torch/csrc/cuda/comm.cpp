#include <torch/csrc/cuda/comm.h>

#ifdef USE_CUDA

#include <torch/csrc/cuda/device_set.h>
#include <torch/csrc/utils/tensor_flatten.h>

#ifdef USE_NCCL
#include <torch/csrc/cuda/nccl.h>
#endif

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/optional.h>

#include <cstddef>
#include <vector>

namespace torch { namespace cuda {
using namespace at;

// Some operations can be performed more efficiently if we're handling tensors
// of a single type only. Adding this logic directly in the loop makes it a bit
// ugly, so here's a helper for it.
struct unique_type_checker {
  void show(const at::Type& t) {
    if (!unique) return;
    if (!type) type = &t;
    unique = (type == &t);
  }

  const at::Type *type = nullptr;
  bool unique = true;
};

std::vector<Tensor> broadcast(const Tensor& tensor, IntList devices) {
  auto & type = tensor.type();
  if (type.is_cuda() && tensor.get_device() != devices[0])
    throw std::runtime_error("device of broadcasted tensor must appear as the "
                             "first on devices list");
  std::vector<Tensor> tensors;
  tensors.reserve(devices.size());
  at::DeviceGuard _device_guard;
#ifdef USE_NCCL
  if (nccl::is_available({tensor})) {
    tensors.push_back(tensor);
    for (auto device : devices.slice(1)) {
      _device_guard.set_index(device);
      tensors.push_back(type.tensor(tensor.sizes()));
    }
    nccl::broadcast(tensors);
  } else {
#else
  {
#endif
    auto & gpu_type = type.toBackend(type.is_sparse() ? at::kSparseCUDA : at::kCUDA);
    if (type.is_cuda()) {
      tensors.push_back(tensor);
    }
    IntList loop_devices = type.is_cuda() ? devices.slice(1) : devices;
    for (auto device : loop_devices) {
      _device_guard.set_index(device);
      tensors.push_back(gpu_type.copy(tensor, true));
    }
  }
  return tensors;
}

tensor_list2d broadcast_coalesced(TensorList tensors, IntList devices, size_t buffer_size) {
  if (!std::all_of(tensors.begin(), tensors.end(),
                   [&](const at::Tensor& t) { return t.get_device() == devices[0]; })) {
    throw std::runtime_error("all tensors must be on devices[0]");
  }

  tensor_list2d outputs(devices.size());
  outputs[0] = tensors;
  for (auto & o : outputs)
    o.reserve(tensors.size());

  unique_type_checker type_checker;
  for (auto & chunk : utils::take_tensors(tensors, buffer_size)) {
    auto & type = chunk.type();
    type_checker.show(type);
    std::vector<at::Tensor> results;
    if (chunk.type().is_sparse()) {
      auto flat_tuple = utils::flatten_sparse_tensors(chunk.tensors);
      std::vector<at::Tensor> broadcast_indices = broadcast(flat_tuple.first, devices);
      std::vector<at::Tensor> broadcast_values = broadcast(flat_tuple.second, devices);
      results.reserve(devices.size());
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        at::DeviceGuard device_guard(devices[i]);
        auto & device_outputs = outputs[i];
        auto & inds = broadcast_indices[i];
        auto & vals = broadcast_values[i];
        for (auto & t : utils::unflatten_sparse_tensors(inds, vals, chunk.tensors))
          device_outputs.push_back(std::move(t));
      }
    } else {
      at::DeviceGuard device_guard(devices[0]);
      std::vector<Tensor> results = broadcast(utils::flatten_dense_tensors(chunk.tensors),
                                              devices);
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.set_index(devices[i]);
        auto & device_outputs = outputs[i];
        for (auto & t : utils::unflatten_dense_tensors(results[i], chunk.tensors))
          device_outputs.push_back(std::move(t));
      }
    }
  }

  // If we only saw a single tensor type, then we can skip expensive reordering
  if (!type_checker.unique) {
    for (auto & o : outputs)
      utils::reorder_tensors_like(o, tensors);
  }
  return outputs;
}

std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntList devices,
    const at::optional<std::vector<int64_t>>& chunk_sizes,
    int64_t dim,
    const at::optional<std::vector<at::cuda::CUDAStream>>& streams) {
  std::vector<at::Tensor> chunks;
  if (chunk_sizes) {
    const int64_t chunk_size_sum =
        std::accumulate(chunk_sizes->begin(), chunk_sizes->end(), 0);
    AT_CHECK(
      chunk_size_sum == tensor.size(dim),
      "given chunk sizes don't sum up to the tensor's size ",
      "(sum(chunk_sizes) == ", chunk_size_sum,
      ", but expected ", tensor.size(dim), ")");
    chunks.reserve(chunk_sizes->size());
    int64_t chunk_start = 0;
    for (size_t chunk = 0; chunk < chunk_sizes->size(); ++chunk) {
      const int64_t chunk_size = (*chunk_sizes)[chunk];
      AT_CHECK(chunk_size > 0, "Chunk size must be positive");
      chunks.push_back(tensor.narrow(dim, chunk_start, chunk_size));
      chunk_start += chunk_size;
    }
    AT_ASSERT(chunks.size() == chunk_sizes->size());
  } else {
    chunks = tensor.chunk(/*chunks=*/devices.size(), /*dim=*/dim);
  }
  at::cuda::CUDAGuard cuda_guard;
  for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
    const auto device_index = static_cast<int32_t>(devices[chunk]);
    if (streams) {
      AT_CHECK(
          (*streams)[chunk].device() == device_index,
          "Expected the device associated with the stream at index ",
          chunk, " (was ", (*streams)[chunk].device(), ") ",
          "to match the device supplied at that index ",
          "(expected ", device_index, ")");
      cuda_guard.set_stream(at::cuda::CUDAStream((*streams)[chunk]));
    }
    chunks[chunk] = chunks[chunk].contiguous().to(
        {at::kCUDA, device_index}, /*non_blocking=*/true);
  }
  return chunks;
}

at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    at::optional<int32_t> destination_index) {
  AT_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
  at::Tensor result;
  int64_t total_size = 0;
  auto& first = tensors.front();
  const auto first_size = first.sizes();
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
  for (const auto& tensor : tensors) {
    AT_CHECK(
        tensor.type().is_cuda(), "Gather expects all inputs to have CUDA type");
    AT_ASSERT(tensor.ndimension() == static_cast<int64_t>(expected_size.size()));
    expected_size[dim] = tensor.size(dim);
    for (size_t dimension = 0; dimension < expected_size.size(); ++dimension) {
      AT_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Gather got an input of invalid size: got ",
          tensor.sizes(), ", but expected ", at::IntList(expected_size));
    }
    total_size += tensor.size(dim);
  }
  expected_size[dim] = total_size;
  at::Device device(at::kCPU);
  if (!destination_index || *destination_index != -1) {
    device = at::Device(at::kCUDA, destination_index ? *destination_index : -1);
  }
  result = at::empty(expected_size, first.options().device(device));

  int64_t chunk_start = 0;
  for (const auto& tensor : tensors) {
    result.narrow(dim, chunk_start, tensor.size(dim))
        .copy_(tensor, /*non_blocking=*/true);
    chunk_start += tensor.size(dim);
  }
  return result;
}
}} // namespace torch::cuda

#endif
