#include "ATen/core/Half.h"
#include "ATen/cuda/CUDAHalf.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace at {
#if CUDA_VERSION < 9000 && !defined(__HIP_PLATFORM_HCC__)
template <> AT_CUDA_API
half convert(Half aten_half) {
  return half{aten_half.x};
}

template <> AT_CUDA_API
half convert(double value) {
  return half{Half(value).x};
}

template <> AT_CUDA_API
Half convert(half cuda_half) {
  return Half(cuda_half.x, Half::from_bits);
}
#else
template <> AT_CUDA_API
half convert(Half aten_half) {
  __half_raw x_raw;
  x_raw.x = aten_half.x;
  return half(x_raw);
}

template <> AT_CUDA_API
Half convert(half cuda_half) {
  __half_raw raw(cuda_half);
  return Half(raw.x, Half::from_bits);
}

template <> AT_CUDA_API
half convert(double value) {
  __half_raw raw;
  raw.x = Half(value).x;
  return half {raw};
}

template <> __half HalfFix(Half h) {
  __half_raw raw;
  raw.x = h.x;
  return __half{raw};
}

template <> Half HalfFix(__half h) {
  __half_raw raw(h);
  return Half(raw.x, Half::from_bits);
}
#endif
} // namespace at
