#include "THTensor.hpp"

#include "generic/THTensor.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THTensor.cpp"
#include "THGenerateHalfType.h"

#include <numeric>

// NB: This is NOT valid on UndefinedTensor
void THTensor_free(THTensor *self)
{
  if (!self) return;
  c10::raw::intrusive_ptr::decref(self);
}

void THTensor_setStorage(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, at::IntList size_, at::IntList stride_) {
  if (stride_.data()) {
    THArgCheck(size_.size() == stride_.size(), 5, "inconsistent size/stride sizes");
  }

#ifdef DEBUG
  THAssert(size_.size() <= INT_MAX);
#endif
  THTensor_setStorageNd(self,
                        storage_,
                        storageOffset_,
                        size_.size(),
                        size_.data(),
                        stride_.data());
}

void THTensor_setStorageNd(THTensor *self, THStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride)
{
  /* storage */
  if(THTensor_getStoragePtr(self) != storage)
  {
    if (!THTensor_getStoragePtr(self)) {
      THError("Tensor: invalid null storage");
    }
    auto scalar_type = at::dataTypeToScalarType(THTensor_getStoragePtr(self)->dtype());
    if(storage)
    {
      c10::raw::intrusive_ptr::incref(storage);
      THTensor_stealAndSetStoragePtr(self, storage);
    }
    else {
      THTensor_stealAndSetStoragePtr(self, THStorage_new(scalar_type));
    }
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
    self->set_storage_offset(storageOffset);

  /* size and stride */
  THTensor_resizeNd(self, nDimension, size, stride);
}

void THTensor_resize(THTensor *self, at::IntList size, at::IntList stride)
{
  if (stride.data()) {
    THArgCheck(stride.size() == size.size(), 3, "invalid stride");
  }

#ifdef DEBUG
  THAssert(size.size() <= INT_MAX);
#endif
  THTensor_resizeNd(self, size.size(), size.data(), stride.data());
}

void THTensor_resizeNd(THTensor *self, int nDimension, const int64_t *size, const int64_t *stride)
{
  AT_CHECK(nDimension >= 0, "resizeNd nDimension must be non-negative");
  int d;
  ptrdiff_t totalSize;
  bool hascorrectsize = true;

  for(d = 0; d < nDimension; d++)
  {
    if((self->dim() > d) && (size[d] != self->size(d))) {
      hascorrectsize = false;
    }

    // NB: this used to test that stride[d] was >= 0
    if((self->dim() > d) && stride && (stride[d] != self->stride(d))) {
      hascorrectsize = false;
    }
  }

  if(nDimension != self->dim()) {
    hascorrectsize = false;
  }

  if(hascorrectsize) {
    return;
  }

  if(nDimension != self->dim())
  {
    self->resize_dim(nDimension);
  }

  totalSize = 1;
  for(d = nDimension-1; d >= 0; d--)
  {
    self->set_size(d, size[d]);
    if(stride && (stride[d] >= 0) ) {
      self->set_stride(d, stride[d]);
    } else {
      if(d == nDimension-1) {
        self->set_stride(d, 1);
      } else {
        // Keep stride monotonically increasing to match NumPy.
        self->set_stride(d, std::max<int64_t>(self->size(d+1), 1)*self->stride(d+1));
      }
    }
    totalSize += (self->size(d)-1)*self->stride(d);
  }

  if(totalSize+self->storage_offset() > 0)
  {
    if(!THTensor_getStoragePtr(self)) {
      THTensor_stealAndSetStoragePtr(self, THStorage_new(self->scalar_type()));
    }
    if(totalSize+self->storage_offset() > THTensor_getStoragePtr(self)->numel()) {
      THStorage_resize(THTensor_getStoragePtr(self), totalSize+self->storage_offset());
    }
  }
}

// On a high level,
// 1. separate oldshape chunks of dimensions, where the dimensions are
//    ``contiguous'' in each chunk, i.e., oldstride[i] = oldshape[i+1] * oldstride[i+1]
// 2. newshape must be able to be separated into same number of chunks as oldshape was separated into,
//    where each chunk of newshape has matching ``numel'', i.e., number of subspaces,
//    as the corresponding chunk of oldshape.
at::optional<std::vector<int64_t>>
THTensor_compute_stride(at::IntList oldshape, at::IntList oldstride, at::IntList newshape) {
  if (oldshape.empty()) {
    return std::vector<int64_t>(newshape.size(), 1);
  }

  // NOTE: stride is arbitrary is somewhat arbitrary in the numel() == 0 case;
  // to match NumPy behavior we copy the strides if the size matches, otherwise
  // we use the stride as if it were computed via resize.
  // This could perhaps be combined with the below code, but the complexity didn't seem worth it.
  int64_t numel = std::accumulate(oldshape.begin(), oldshape.end(), 1, std::multiplies<int64_t>());
  if (numel == 0 && oldshape.equals(newshape)) {
    return oldstride.vec();
  }

  std::vector<int64_t> newstride(newshape.size());
  if (numel == 0) {
    int64_t view_numel = 1;
    for (int64_t view_d = newshape.size() - 1; view_d >= 0; view_d--) {
      if (view_d == newshape.size() - 1) {
        newstride[view_d] = 1;
      } else {
        newstride[view_d] = std::max<int64_t>(newshape[view_d+1], 1) * newstride[view_d+1];
      }
    }
    return newstride;
  }

  int64_t view_d = newshape.size() - 1;
  // stride for each subspace in the chunk
  int64_t chunk_base_stride = oldstride.back();
  // numel in current chunk
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;
  for (int64_t tensor_d = oldshape.size() - 1; tensor_d >= 0; tensor_d--) {
    tensor_numel *= oldshape[tensor_d];
    // if end of tensor size chunk, check view
    if ((tensor_d == 0) ||
        (oldshape[tensor_d - 1] != 1 && oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
      while (view_d >= 0 && (view_numel < tensor_numel || newshape[view_d] == 1)) {
        newstride[view_d] = view_numel * chunk_base_stride;
        view_numel *= newshape[view_d];
        view_d--;
      }
      if (view_numel != tensor_numel) {
        return at::nullopt;
      }
      if (tensor_d > 0) {
        chunk_base_stride = oldstride[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }
  if (view_d != -1) {
    return at::nullopt;
  }
  return newstride;
}

// NB: Steals ownership of storage
void THTensor_stealAndSetStoragePtr(THTensor* tensor, THStorage* storage) {
  // Caffe2 might have tensors whose storages are null, but we
  // don't allow it in PyTorch.
  AT_ASSERT(storage);
  tensor->storage_ = at::Storage(storage);
}
