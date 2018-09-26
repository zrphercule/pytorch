#pragma once

#include "ATen/core/ATenGeneral.h"
#include "ATen/Context.h"
#include "ATen/cuda/CUDAStream.h"
#include "ATen/cuda/Exceptions.h"

#include <cstdint>

#include "cuda_runtime_api.h"
#include "cusparse.h"
#include "cublas_v2.h"

namespace at {
namespace cuda {

/*
A common CUDA interface for ATen.

This interface is distinct from CUDAHooks, which defines an interface that links
to both CPU-only and CUDA builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
CUDA builds.

CUDAContext, on the other hand, should be preferred by files only included in
CUDA builds. It is intended to expose CUDA functionality in a consistent
manner.

This means there is some overlap between the CUDAContext and CUDAHooks, but
the choice of which to use is simple: use CUDAContext when in a CUDA-only file,
use CUDAHooks otherwise.

Note that CUDAContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single CUDA context/state.
*/

/* Device info */
CAFFE2_API int64_t getNumGPUs();

CAFFE2_API int64_t current_device();

CAFFE2_API void set_device(int64_t device);

CAFFE2_API cudaDeviceProp* getCurrentDeviceProperties();

CAFFE2_API cudaDeviceProp* getDeviceProperties(int64_t device);

/* Streams */
CAFFE2_API CUDAStream
createCUDAStream(const bool isHighPriority = false, int64_t device = -1);

CAFFE2_API CUDAStream getDefaultCUDAStream(int64_t device = -1);
CAFFE2_API CUDAStream getCurrentCUDAStream(int64_t device = -1);

CAFFE2_API void setCurrentCUDAStream(CUDAStream stream);
CAFFE2_API void uncheckedSetCurrentCUDAStream(CUDAStream stream);

CAFFE2_API Allocator* getCUDADeviceAllocator();

/* Handles */
#ifndef __HIP_PLATFORM_HCC__
CAFFE2_API cusparseHandle_t getCurrentCUDASparseHandle();
CAFFE2_API cublasHandle_t getCurrentCUDABlasHandle();
#endif


} // namespace cuda
} // namespace at
