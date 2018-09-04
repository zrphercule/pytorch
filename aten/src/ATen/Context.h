#pragma once

#include <ATen/CPUGeneral.h>
#include "ATen/core/ATenGeneral.h"
#include "ATen/CUDAStream.h"
#include "ATen/core/Generator.h"
#include "ATen/Type.h"
#include "ATen/Utils.h"
#include "ATen/core/Error.h"
#include "ATen/detail/CUDAHooksInterface.h"
#include "ATen/detail/VariableHooksInterface.h"

// This is temporary
#include "ATen/core/ATenCoreTest.h"

#include <memory>
#include <mutex>
#include <cstdint>

namespace at {

class AT_API Context {
public:
  Context();
  Type* getNonVariableTypeRaw(Backend p, ScalarType s) {
    return type_registry[static_cast<int>(p)][static_cast<int>(s)].get();
  }
  Type * getNonVariableTypeOpt(Backend p, ScalarType s) {
    if (p != Backend::Undefined) initCUDAIfNeeded(backendToDeviceType(p));
    auto type = getNonVariableTypeRaw(p, s);

    if(!type) {
      // there is only a single Undefined Type.
      if (p == Backend::Undefined || s == ScalarType::Undefined) {
        return getNonVariableTypeRaw(Backend::Undefined, ScalarType::Undefined);
      }
    }

    return type;
  }
  Type & getNonVariableType(Backend p, ScalarType s) {
    auto* type = getNonVariableTypeOpt(p, s);
    if (!type) AT_ERROR(toString(p), toString(s), "Type is not enabled.");
    return *type;
  }
  Type & getVariableType(Backend p, ScalarType s) {
    auto& baseType = getNonVariableType(p, s);
    return detail::getVariableHooks().getVariableTypeFromBaseType(baseType);
  }
  Type & getMaybeVariableType(Backend p, ScalarType s, bool is_variable) {
    if (is_variable) {
      return getVariableType(p, s);
    } else {
      return getNonVariableType(p, s);
    }
  }

  Generator & defaultGenerator(DeviceType device_type) {
    initCUDAIfNeeded(device_type);
    auto & generator = generator_registry[static_cast<int>(device_type)];
    if(!generator)
      AT_ERROR(DeviceTypeName(device_type), " backend type not enabled.");
    return *generator;
  }
  bool hasMKL() const;
  bool hasLAPACK() const;
  bool hasMAGMA() const {
    return detail::getCUDAHooks().hasMAGMA();
  }
  bool hasCUDA() const {
    return detail::getCUDAHooks().hasCUDA();
  }
  bool hasCuDNN() const {
    return detail::getCUDAHooks().hasCuDNN();
  }
  int64_t current_device() const {
    return detail::getCUDAHooks().current_device();
  }
  // defined in header so that getNonVariableType has ability to inline
  // call_once check. getNonVariableType is called fairly frequently
  THCState* lazyInitCUDA() {
    std::call_once(thc_init,[&] {
      thc_state = detail::getCUDAHooks().initCUDA();
      generator_registry[static_cast<int>(DeviceType::CUDA)] =
        detail::getCUDAHooks().initCUDAGenerator(this);
      detail::getCUDAHooks().registerCUDATypes(this);
    });
    return thc_state.get();
  }

  THCState* getTHCState() {
    // AT_ASSERT(thc_state);
    return thc_state.get();
  }

  int getNumGPUs() const {
    return detail::getCUDAHooks().getNumGPUs();
  }
  size_t freshTypeID() {
    return next_id++;
  }
  bool setFlushDenormal(bool on);

  // NB: This method is *purely* whether or not a user requested
  // that CuDNN was enabled, it doesn't actually say anything about
  // whether or not CuDNN is actually usable.  Use cudnn_is_acceptable
  // to test this instead
  bool userEnabledCuDNN() const;
  void setUserEnabledCuDNN(bool e);
  bool benchmarkCuDNN() const;
  void setBenchmarkCuDNN(bool);
  bool deterministicCuDNN() const;
  void setDeterministicCuDNN(bool);
  std::unique_ptr<Generator>
    generator_registry[static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];
private:
  // NB: type_registry has nullptr for all CUDA backends until
  // CUDA initialization has occurred
  std::unique_ptr<Type> type_registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
  void initCUDAIfNeeded(DeviceType p) {
    if (p == DeviceType::CUDA) {
      lazyInitCUDA();
    }
  }
  std::once_flag thc_init;
  bool enabled_cudnn = true;
  bool deterministic_cudnn = false;
  bool benchmark_cudnn = false;
  std::atomic<size_t> next_id;
  std::unique_ptr<THCState, void(*)(THCState*)> thc_state;
  friend struct Type;
  friend void register_cpu_types(Context * context);
  friend void register_cuda_types(Context * context);
};

AT_API Context & globalContext();

static inline void init() {
  globalContext();
  if (const char *env_p = std::getenv("OMP_NUM_THREADS")) {
    at::set_num_threads(std::stoi(env_p));
  }
  if (const char *env_p = std::getenv("MKL_NUM_THREADS")) {
    at::set_num_threads(std::stoi(env_p));
  }
}

static inline Type& getNonVariableType(Backend p, ScalarType s) {
  return globalContext().getNonVariableType(p, s);
}

static inline Type& getNonVariableType(DeviceType p, ScalarType s) {
  return globalContext().getNonVariableType(deviceTypeToBackend(p), s);
}

AT_API Type& getMaybeVariableType(TensorOptions options);
AT_API Type& getMaybeVariableType(const TensorImpl*);

static inline Type& CPU(ScalarType s) {
  return getNonVariableType(Backend::CPU, s);
}

static inline Type& CUDA(ScalarType s) {
  return getNonVariableType(Backend::CUDA, s);
}

static inline bool hasCUDA() {
  return globalContext().hasCUDA();
}

static inline bool hasCuDNN() {
  return globalContext().hasCuDNN();
}

static inline bool hasMKL() {
  return globalContext().hasMKL();
}

static inline bool hasLAPACK() {
  return globalContext().hasLAPACK();
}

static inline bool hasMAGMA() {
  return globalContext().hasMAGMA();
}

static inline int64_t current_device() {
  return globalContext().current_device();
}

} // namespace at
