#include "catch_utils.hpp"

#include <torch/tensor.h>

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/OptionsGuard.h>
#include <ATen/core/TensorOptions.h>

#include <vector>
#include <string>

using namespace at;

// A macro so we don't lose location information when an assertion fails.
#define REQUIRE_OPTIONS(device_, index_, type_, layout_)                    \
  CATCH_REQUIRE(options.device().type() == Device((device_), (index_)).type());   \
  CATCH_REQUIRE(options.device().index() == Device((device_), (index_)).index()); \
  CATCH_REQUIRE(options.dtype() == (type_));                                      \
  CATCH_REQUIRE(options.layout() == (layout_))

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  CATCH_REQUIRE(tensor.device().type() == Device((device_), (index_)).type());   \
  CATCH_REQUIRE(tensor.device().index() == Device((device_), (index_)).index()); \
  CATCH_REQUIRE(tensor.type().scalarType() == (type_));                          \
  CATCH_REQUIRE(tensor.type().layout() == (layout_))

CATCH_TEST_CASE("TensorOptions/DefaultsToTheRightValues") {
  TensorOptions options;
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
}

CATCH_TEST_CASE("TensorOptions/ReturnsTheCorrectType") {
  auto options = TensorOptions().device(kCPU).dtype(kInt).layout(kSparse);
  CATCH_REQUIRE(at::getType(options) == getNonVariableType(Backend::SparseCPU, kInt));
}

CATCH_TEST_CASE("TensorOptions/UtilityFunctionsReturnTheRightTensorOptions") {
  auto options = dtype(kInt);
  REQUIRE_OPTIONS(kCPU, -1, kInt, kStrided);

  options = layout(kSparse);
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kSparse);

  options = device({kCUDA, 1});
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = device_index(1);
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = dtype(kByte).layout(kSparse).device({kCUDA, 2}).device_index(3);
  REQUIRE_OPTIONS(kCUDA, 3, kByte, kSparse);
}

CATCH_TEST_CASE("TensorOptions/ConstructsWellFromCPUTypes") {
  TensorOptions options;
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);

  options = TensorOptions({kCPU, 0});
  REQUIRE_OPTIONS(kCPU, 0, kFloat, kStrided);

  options = TensorOptions(kInt);
  REQUIRE_OPTIONS(kCPU, -1, kInt, kStrided);

  options = TensorOptions(getNonVariableType(Backend::SparseCPU, kFloat));
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kSparse);

  options = TensorOptions(getNonVariableType(Backend::SparseCPU, kByte));
  REQUIRE_OPTIONS(kCPU, -1, kByte, kSparse);
}

CATCH_TEST_CASE("TensorOptions/ConstructsWellFromCPUTensors") {
  auto options = empty(5, kDouble).options();
  REQUIRE_OPTIONS(kCPU, -1, kDouble, kStrided);

  options = empty(5, getNonVariableType(Backend::SparseCPU, kByte)).options();
  REQUIRE_OPTIONS(kCPU, -1, kByte, kSparse);
}

CATCH_TEST_CASE("TensorOptions/ConstructsWellFromVariables") {
  auto options = torch::empty(5).options();
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
  CATCH_REQUIRE(!options.requires_grad());

  options = torch::empty(5, at::requires_grad()).options();
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
  CATCH_REQUIRE(!options.requires_grad());
}

CATCH_TEST_CASE("Device/ParsesCorrectlyFromString") {
  Device device("cpu:0");
  CATCH_REQUIRE(device == Device(kCPU, 0));

  device = Device("cpu");
  CATCH_REQUIRE(device == Device(kCPU));

  device = Device("cuda:123");
  CATCH_REQUIRE(device == Device(kCUDA, 123));

  device = Device("cuda");
  CATCH_REQUIRE(device == Device(kCUDA));

  std::vector<std::string> badnesses = {
      "", "cud:1", "cuda:", "cpu::1", ":1", "3", "tpu:4", "??"};
  for (const auto& badness : badnesses) {
    _CATCH_REQUIRE_THROWS(Device(badness));
  }
}

CATCH_TEST_CASE("OptionsGuard") {
  Tensor tensor;
  {
    OptionsGuard guard(TensorOptions{});
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kFloat, kStrided);

  {
    OptionsGuard guard(TensorOptions().dtype(kInt));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kInt, kStrided);

  {
    OptionsGuard guard(TensorOptions().dtype(kInt).layout(kSparse));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kInt, kSparse);

  {
    OptionsGuard guard(requires_grad(true));
    tensor = torch::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kFloat, kStrided);
  CATCH_REQUIRE(tensor.requires_grad());
}
