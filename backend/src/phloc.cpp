#include "phloc.hpp"
#include <iostream>

namespace c10d {


bool WorkPhloc::isCompleted() {
  return true;
}

bool WorkPhloc::isSuccess() const {
  return true;
}

bool WorkPhloc::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkPhloc::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
BackendPhloc::BackendPhloc(int rank, int size)
    : Backend(rank, size) {}

// This is a Phloc allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendPhloc::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  for (auto& outputTensorVec : outputTensors) {
      for (auto& outputTensor : outputTensorVec) {
          outputTensor.zero_();
      }
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkPhloc>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendPhloc::_allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

// This is a Phloc allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendPhloc::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  for (auto& tensor : tensors) {
      tensor.zero_();
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkPhloc>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendPhloc::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::barrier(
    const BarrierOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendPhloc::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Backend> BackendPhloc::createBackendPhloc(
    const c10::intrusive_ptr<::c10d::Store>& /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<BackendPhloc>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createBackendPhloc", &BackendPhloc::createBackendPhloc);
}

} // namespace c10d