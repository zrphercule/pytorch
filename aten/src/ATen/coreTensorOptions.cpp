#include <ATen/core/TensorOptions.h>

#include <ATen/Device.h>
#include <ATen/core/Layout.h>
#include <ATen/OptionsGuard.h>
#include <ATen/ScalarType.h>
#include <ATen/core/optional.h>
#include <ATen/core/ScalarType.h>

#include <iostream>

namespace at {

TensorOptions::TensorOptions(bool use_thread_local_default_options) {
  if (use_thread_local_default_options) {
    this->dtype(DefaultTensorOptions::get().dtype());
    this->device(DefaultTensorOptions::get().device());
    this->layout(DefaultTensorOptions::get().layout());
    this->requires_grad(DefaultTensorOptions::get().requires_grad());
  }
}

std::ostream& operator<<(
    std::ostream& stream,
    const TensorOptions& options) {
  return stream << "TensorOptions(dtype=" << options.dtype()
                << ", device=" << options.device()
                << ", layout=" << options.layout()
                << ", requires_grad=" << std::boolalpha
                << options.requires_grad() << ")";
}

} // namespace at
