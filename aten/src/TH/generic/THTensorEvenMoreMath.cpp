#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorEvenMoreMath.cpp"
#else

#include <TH/generic/THTensorApply.hpp>

void THTensor_(fill)(THTensor *r_, real value)
{
  if (THTensor_(isContiguous)(r_) || THTensor_(isTransposed)(r_)) {
    TH_TENSOR_APPLY_CONTIG(real, r_, THVector_(fill)(r__data, value, r__len););
  } else {
    TH_TENSOR_APPLY(real, r_,
      if (r__stride == 1) {
        THVector_(fill)(r__data, value, r__size);
	r__i = r__size;
	r__data += r__stride * r__size;
	break;
      } else {
        *r__data = value;
      }
      );
  }
}

void THTensor_(zero)(THTensor *r_)
{
  THTensor_(fill)(r_, 0);
}

void THTensor_(maskedFill)(THTensor *tensor, THByteTensor *mask, real value)
{
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THFree(mask_counter);
                     THFree(tensor_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     *tensor_data = value;
                   });
}

void THTensor_(maskedCopy)(THTensor *tensor, THByteTensor *mask, THTensor* src )
{
  THTensor *srct = THTensor_(newContiguous)(src);
  real *src_data = THTensor_(data)(srct);
  ptrdiff_t cntr = 0;
  ptrdiff_t nelem = THTensor_(nElement)(srct);
  if (THTensor_(nElement)(tensor) != THByteTensor_nElement(mask))
  {
    THTensor_(free)(srct);
    THError("Number of elements of destination tensor != Number of elements in mask");
  }
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THTensor_(free)(srct);
                     THFree(mask_counter);
                     THFree(tensor_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     if (cntr == nelem)
                     {
                       THTensor_(free)(srct);
                       THFree(mask_counter);
                       THFree(tensor_counter);
                       THError("Number of elements of src < number of ones in mask");
                     }
                     *tensor_data = *src_data;
                     src_data++;
                     cntr++;
                   });
  THTensor_(free)(srct);
}

void THTensor_(maskedSelect)(THTensor *tensor, THTensor *src, THByteTensor *mask)
{
  ptrdiff_t numel = THByteTensor_sumall(mask);
  real *tensor_data;

#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THTensor_(resize1d)(tensor,numel);
  tensor_data = THTensor_(data)(tensor);
  TH_TENSOR_APPLY2(real, src, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THFree(mask_counter);
                     THFree(src_counter);
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     *tensor_data = *src_data;
                     tensor_data++;
                   });
}

// Finds non-zero elements of a tensor and returns their subscripts
void THTensor_(nonzero)(THLongTensor *subscript, THTensor *tensor)
{
  ptrdiff_t numel = 0;
  int64_t *subscript_data;
  int64_t i = 0;
  int64_t dim;
  int64_t div = 1;
#ifdef TH_REAL_IS_HALF
#define IS_NONZERO(val) ((val.x & 0x7fff) != 0)
#else
#define IS_NONZERO(val) ((val)!=0)
#endif

  /* First Pass to determine size of subscripts */
  TH_TENSOR_APPLY(real, tensor,
                  if IS_NONZERO(*tensor_data) {
                    ++numel;
                  });
#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THLongTensor_resize2d(subscript, numel, tensor->dim());

  /* Second pass populates subscripts */
  subscript_data = THLongTensor_data(subscript);
  TH_TENSOR_APPLY(real, tensor,
                  if IS_NONZERO(*tensor_data) {
                    div = 1;

                    for (dim = tensor->dim() - 1; dim >= 0; dim--) {
                      *(subscript_data + dim) = (i/div) % tensor->size(dim);
                      div *= tensor->size(dim);
                    }

                    subscript_data += tensor->dim();
                  }
                  ++i;);
}

void THTensor_(indexSelect)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
  ptrdiff_t i, numel;
  THLongStorage *newSize;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;
  real *tensor_data, *src_data;

#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(index->_dim() <= 1, 3, "Index is supposed to be an empty tensor or a vector");
  THArgCheck(dim < src->_dim(), 4, "Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  THArgCheck(src->_dim() > 0, 2, "Source tensor is empty");
#else
  THArgCheck(index->dim() == 1, 3, "Index is supposed to be 1-dimensional");
  THArgCheck(dim < src->dim(), 4, "Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
  //THArgCheck(src->dim() > 0, 2, "Source tensor is empty");
#endif

  numel = THLongTensor_nElement(index);

  newSize = THLongStorage_newWithSize(src->dim());
  THLongStorage_rawCopy(newSize, THTensor_getSizePtr(src));
#ifdef DEBUG
  THAssert(numel <= LONG_MAX);
#endif
  THLongStorage_data(newSize)[dim] = numel;
  THTensor_(resize)(tensor,newSize,NULL);
  THLongStorage_free(newSize);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (dim == 0 && THTensor_(isContiguous)(src) && THTensor_(isContiguous)(tensor))
  {
    tensor_data = THTensor_(data)(tensor);
    src_data = THTensor_(data)(src);
    ptrdiff_t rowsize = src->size(0) == 0 ? 1: THTensor_(nElement)(src) / src->size(0);

    // check that the indices are within range
    int64_t max = src->size(0) - 1 + TH_INDEX_BASE;
    for (i=0; i<numel; i++) {
      if (index_data[i] < TH_INDEX_BASE || index_data[i] > max) {
        THLongTensor_free(index);
        THError("index out of range");
      }
    }

    if (src->dim() == 1) {
      #pragma omp parallel for if(numel > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<numel; i++)
        tensor_data[i] = src_data[index_data[i] - TH_INDEX_BASE];
    } else {
      #pragma omp parallel for if(numel*rowsize > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<numel; i++)
        memcpy(tensor_data + i*rowsize, src_data + (index_data[i] - TH_INDEX_BASE)*rowsize, rowsize*sizeof(real));
    }
  }
  else if (src->dim() == 1)
  {
    for (i=0; i<numel; i++)
      THTensor_(set1d)(tensor,i,THTensor_(get1d)(src,index_data[i] - TH_INDEX_BASE));
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      tSlice = THTensor_(new)();
      sSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor, dim, i);
      THTensor_(select)(sSlice, src, dim, index_data[i] - TH_INDEX_BASE);
      THTensor_(copy)(tSlice, sSlice);
      THTensor_(free)(tSlice);
      THTensor_(free)(sSlice);
    }
  }

  THLongTensor_free(index);
}

void THTensor_(indexCopy)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  ptrdiff_t i, numel;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;

  // Error checking for this function has moved to ATen!!

  numel = THLongTensor_nElement(index);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->dim() > 1 )
  {
    tSlice = THTensor_(new)();
    sSlice = THTensor_(new)();

    for (i=0; i<numel; i++)
    {
      THTensor_(select)(tSlice, tensor, dim, index_data[i] - TH_INDEX_BASE);
      THTensor_(select)(sSlice, src, dim, i);
      THTensor_(copy)(tSlice, sSlice);
    }

    THTensor_(free)(tSlice);
    THTensor_(free)(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THTensor_(set1d)(tensor, index_data[i] - TH_INDEX_BASE, THTensor_(get1d)(src,i));
    }
  }
  THLongTensor_free(index);
}

static ptrdiff_t THTensor_(dataOffset)(THTensor* tensor, ptrdiff_t linearIndex) {
  auto size = tensor->sizes();
  auto stride = tensor->strides();
  int nDim = tensor->_dim();
  ptrdiff_t dataOffset = 0;
  for (int i = nDim - 1; i >= 0; i--) {
    dataOffset += (linearIndex % size[i]) * stride[i];
    linearIndex /= size[i];
  }
  return dataOffset;
}

static inline void THTensor_(checkLinearIndex)(int64_t linearIndex, int64_t numel) {
  THArgCheck(linearIndex < numel && linearIndex >= -numel, 2, "out of range: %d out of %d", (int)linearIndex, (int)numel);
}

static inline int64_t THTensor_(wrapLinearIndex)(int64_t linearIndex, int64_t numel) {
  return linearIndex < 0 ? linearIndex + numel : linearIndex;
}

void THTensor_(take)(THTensor *r_, THTensor *src, THLongTensor *index)
{
  THTensor_(resizeNd)(r_, index->dim(), THTensor_getSizePtr(index), NULL);
  THTensor* dst = THTensor_(newContiguous)(r_);

  index = THLongTensor_newContiguous(index);
  int64_t* index_data = THLongTensor_data(index);
  ptrdiff_t srcElements = THTensor_(nElement)(src);
  real* src_data = THTensor_(data)(src);
  real* dst_data = THTensor_(data)(dst);
  ptrdiff_t nIndices = THLongTensor_nElement(index);
  int isContiguous = THTensor_(isContiguous)(src);

  // Exceptions must not be thrown across OpenMP parallel sections, so we
  // record the position of the invalid index and throw the exception after the
  // loop.
  std::atomic<int64_t> invalidIdxPos(-1);

  ptrdiff_t i;
  #pragma omp parallel for if(nIndices > TH_OMP_OVERHEAD_THRESHOLD) private(i)
  for (i = 0; i < nIndices; i++) {
    int64_t idx = index_data[i];
    if (idx < srcElements && idx >= -srcElements) {
      idx = THTensor_(wrapLinearIndex)(idx, srcElements);
      if (isContiguous) {
        dst_data[i] = src_data[idx];
      } else {
        dst_data[i] = src_data[THTensor_(dataOffset)(src, idx)];
      }
    } else {
      int64_t tmp = -1;
      invalidIdxPos.compare_exchange_strong(tmp, i);
    }
  }

  if (invalidIdxPos >= 0) {
    THTensor_(checkLinearIndex)(index_data[invalidIdxPos], srcElements);
  }

  THLongTensor_free(index);
  THTensor_(freeCopyTo)(dst, r_);
}

void THTensor_(put)(THTensor *tensor, THLongTensor *index, THTensor *src, int accumulate)
{
  THArgCheck(THLongTensor_nElement(index) == THTensor_(nElement)(src), 3,
    "src should have the same number of elements as index");

  index = THLongTensor_newContiguous(index);
  src = THTensor_(newContiguous)(src);
  real* data = THTensor_(data)(tensor);
  ptrdiff_t numel = THTensor_(nElement)(tensor);
  int is_contiguous = THTensor_(isContiguous)(tensor);

  TH_TENSOR_APPLY2(int64_t, index, real, src,
    THTensor_(checkLinearIndex)(*index_data, numel);
    int64_t linearIndex = THTensor_(wrapLinearIndex)(*index_data, numel);
    int64_t dataOffset = is_contiguous ? linearIndex : THTensor_(dataOffset)(tensor, linearIndex);
    if (accumulate) {
      data[dataOffset] += *src_data;
    } else {
      data[dataOffset] = *src_data;
    }
  );

  THTensor_(free)(src);
  THLongTensor_free(index);
}

void THTensor_(indexAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  ptrdiff_t i, numel;
  THTensor *tSlice, *sSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(index->_dim() == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->_dim(), 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
#else
  THArgCheck(index->dim() == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->dim(), 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
#endif
  THArgCheck(numel == src->size(dim),4,"Number of indices should be equal to source:size(dim)");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  if (tensor->dim() > 1)
  {
    tSlice = THTensor_(new)();
    sSlice = THTensor_(new)();

    for (i=0; i<numel; i++)
    {
      THTensor_(select)(tSlice, tensor, dim, index_data[i] - TH_INDEX_BASE);
      THTensor_(select)(sSlice, src, dim, i);
      THTensor_(cadd)(tSlice, tSlice, 1.0, sSlice);
    }

    THTensor_(free)(tSlice);
    THTensor_(free)(sSlice);
  }
  else
  {
    for (i=0; i<numel; i++)
    {
      THTensor_(set1d)(tensor,
              index_data[i] - TH_INDEX_BASE,
              THTensor_(get1d)(src,i) + THTensor_(get1d)(tensor,index_data[i] - TH_INDEX_BASE));
    }
  }
  THLongTensor_free(index);
}

void THTensor_(indexFill)(THTensor *tensor, int dim, THLongTensor *index, real val)
{
  ptrdiff_t i, numel;
  THTensor *tSlice;
  int64_t *index_data;

  numel = THLongTensor_nElement(index);
#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(index->_dim() == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->_dim(), 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
#else
  THArgCheck(index->dim() == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->dim(), 4,"Indexing dim %d is out of bounds of tensor", dim + TH_INDEX_BASE);
#endif

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  for (i=0; i<numel; i++)
  {
    if (tensor->dim() > 1)
    {
      tSlice = THTensor_(new)();
      THTensor_(select)(tSlice, tensor,dim,index_data[i] - TH_INDEX_BASE);
      THTensor_(fill)(tSlice, val);
      THTensor_(free)(tSlice);
    }
    else
    {
      THTensor_(set1d)(tensor, index_data[i] - TH_INDEX_BASE, val);
    }
  }
  THLongTensor_free(index);
}

void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(src), 4,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(dim >= 0 && dim < THTensor_(nDimension)(tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THTensor_(nDimension)(src) == THTensor_(nDimension)(tensor), 2,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(real, tensor, real, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= src_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in gather");
                         }
                         *(tensor_data + i*tensor_stride) = src_data[(idx - TH_INDEX_BASE) * src_stride];
                       })
}

void THTensor_(scatter)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  int64_t elems_per_row, i, idx;

#ifndef USE_TH_SIZE_ZERO_DIM
  THArgCheck(dim < THTensor_(_nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor__nDimension(index) == THTensor_(_nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THTensor_(_nDimension)(src) == THTensor_(_nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");
#else
  THArgCheck(dim < THTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THTensor_(nDimension)(src) == THTensor_(nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");
#endif

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(real, tensor, real, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_SCATTER,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= tensor_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatter");
                         }
                         tensor_data[(idx - TH_INDEX_BASE) * tensor_stride] = *(src_data + i*src_stride);
                       })
}

void THTensor_(scatterAdd)(THTensor *tensor, int dim, THLongTensor *index, THTensor *src)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THTensor_(nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor_nDimension(index) == THTensor_(nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(THTensor_(nDimension)(src) == THTensor_(nDimension)(tensor), 4,
             "Input tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY3(real, tensor, real, src, int64_t, index, dim,
                       TH_TENSOR_DIM_APPLY3_SIZE_SCATTER,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= tensor_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatterAdd");
                         }
                         tensor_data[(idx - TH_INDEX_BASE) * tensor_stride] += *(src_data + i*src_stride);
                       })
}

void THTensor_(scatterFill)(THTensor *tensor, int dim, THLongTensor *index, real val)
{
  int64_t elems_per_row, i, idx;

  THArgCheck(dim < THTensor_(_nDimension)(tensor), 2, "Index dimension is out of bounds");
  THArgCheck(THLongTensor__nDimension(index) == THTensor_(_nDimension)(tensor), 3,
             "Index tensor must have same dimensions as output tensor");

  elems_per_row = THLongTensor_size(index, dim);

  TH_TENSOR_DIM_APPLY2(real, tensor, int64_t, index, dim,
                       for (i = 0; i < elems_per_row; ++i)
                       {
                         idx = *(index_data + i*index_stride);
                         if (idx < TH_INDEX_BASE || idx >= tensor_size + TH_INDEX_BASE)
                         {
                           THFree(TH_TENSOR_DIM_APPLY_counter);
                           THError("Invalid index in scatter");
                         }
                         tensor_data[(idx - TH_INDEX_BASE) * tensor_stride] = val;
                       })
}

accreal THTensor_(dot)(THTensor *tensor, THTensor *src)
{
  accreal sum = 0;
  /* we use a trick here. careful with that. */
  TH_TENSOR_APPLY2(real, tensor, real, src,
                   int64_t sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   sum += THBlas_(dot)(sz, src_data, src_stride, tensor_data, tensor_stride);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride;
                   break;);
  return sum;
}

real THTensor_(minall)(THTensor *tensor)
{
  real theMin;
  real value;

  THArgCheck(tensor->_dim() > 0, 1, "tensor must have one dimension");
  theMin = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor,
                  value = *tensor_data;
                  /* This is not the same as value<theMin in the case of NaNs */
                  if(!(value >= theMin))
                  {
                    theMin = value;
                    th_isnan_break(value)
                  });
  return theMin;
}

real THTensor_(maxall)(THTensor *tensor)
{
  real theMax;
  real value;

  THArgCheck(tensor->_dim() > 0, 1, "tensor must have one dimension");
  theMax = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor,
                  value = *tensor_data;
                  /* This is not the same as value>theMax in the case of NaNs */
                  if(!(value <= theMax))
                  {
                    theMax = value;
                    th_isnan_break(value)
                  });
  return theMax;
}

accreal THTensor_(sumall)(THTensor *tensor)
{
  accreal sum = 0;
  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
  if(inOMP) {
    serial_path = 1;
  } else {
    TH_TENSOR_APPLY_REDUCTION_OMP(real, tensor, +:sum, sum += *tensor_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
  }
#else
    serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY(real, tensor, sum += *tensor_data;);
  }
  return sum;
}

accreal THTensor_(prodall)(THTensor *tensor)
{
  accreal prod = 1;
  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
  if(inOMP) {
    serial_path = 1;
  } else {
    TH_TENSOR_APPLY_REDUCTION_OMP(real, tensor, *:prod, prod *= *tensor_data;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
  }
#else
    serial_path = 1;
#endif
  if (serial_path) {
    TH_TENSOR_APPLY(real, tensor, prod *= *tensor_data;);
  }
  return prod;
}

void THTensor_(add)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(adds)(r__data, t_data, value, r__len););
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data + value;, ORDIN_TH_OMP_OVERHEAD_THRESHOLD)
    }
#else
    (void)r_Size;
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}

void THTensor_(sub)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(add)(r_, t, -value);
}

void THTensor_(add_scaled)(THTensor *r_, THTensor *t, real value, real alpha)
{
  THTensor_(add)(r_, t, value * alpha);
}

void THTensor_(sub_scaled)(THTensor *r_, THTensor *t, real value, real alpha)
{
  THTensor_(add)(r_, t, -value * alpha);
}

void THTensor_(mul)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(muls)(r__data, t_data, value, r__len););
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data * value;, ORDIN_TH_OMP_OVERHEAD_THRESHOLD)
    }
#else
    (void)r_Size;
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * value;);
  }
}

void THTensor_(div)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(divs)(r__data, t_data, value, r__len););
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data / value;, ORDIN_TH_OMP_OVERHEAD_THRESHOLD)
    }
#else
    (void)r_Size;
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data / value;);
  }
}

void THTensor_(lshift)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT)
  return THTensor_(mul)(r_, t, powf(2, value));
#elif defined(TH_REAL_IS_DOUBLE)
  return THTensor_(mul)(r_, t, pow(2, value));
#elif defined(TH_REAL_IS_HALF)
  return THError("lshift is not supported for torch.HalfTensor");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
    for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_BYTE)
      rp[i] = ((real) tp[i]) << value;
#else
      rp[i] = ((ureal) tp[i]) << value;
#endif
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
#if defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (((real) *t_data) << value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (((ureal) *t_data) << value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
#if defined(TH_REAL_IS_BYTE)
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((real) *t_data) << value););
#else
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((ureal) *t_data) << value););
#endif
  }
#endif
}

void THTensor_(rshift)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT)
  return THTensor_(div)(r_, t, powf(2, value));
#elif defined(TH_REAL_IS_DOUBLE)
  return THTensor_(div)(r_, t, pow(2, value));
#elif defined(TH_REAL_IS_HALF)
  return THError("rshift is not supported for torch.HalfTensor");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
    for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_BYTE)
      rp[i] = ((real) tp[i]) >> value;
#else
      rp[i] = ((ureal) tp[i]) >> value;
#endif
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
#if defined(TH_REAL_IS_BYTE)
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (((real) *t_data) >> value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (((ureal) *t_data) >> value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
#if defined(TH_REAL_IS_BYTE)
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((real) *t_data) >> value););
#else
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (((ureal) *t_data) >> value););
#endif
  }
#endif
}

void THTensor_(fmod)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      rp[i] = fmod(tp[i], value);
#else
      rp[i] = tp[i] % value;
#endif
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = fmod(*t_data, value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (*t_data % value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = fmod(*t_data, value););
#else
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (*t_data % value););
#endif
  }
}

// Should wrap if the value (a) has a different sign than the divisor (b), but is not 0.
static inline bool modulo_wrap(real a, real b) {
  return (a != 0) && (a < 0) != (b < 0);
}

void THTensor_(remainder)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int tContig = THTensor_(isContiguous)(t);
  int serial_path = 0;
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD) private(i)
    for (i=0; i<r_Size; i++) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      rp[i] = (value == 0)? NAN : tp[i] - value * floor(tp[i] / value);
#else
      // There is no NAN for integers
      rp[i] = tp[i] % value;
      if (modulo_wrap(rp[i], value))
        rp[i] += value;
#endif
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = (value == 0)? NAN : *t_data - value * floor(*t_data / value);, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#else
      // There is no NAN for integers
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data % value;
                                        if (modulo_wrap(*r__data, value)) *r__data += value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
#endif
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (value == 0)? NAN : *t_data - value * floor(*t_data / value););
#else
    // There is no NAN for integers
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data % value;
                                          if (modulo_wrap(*r__data, value)) *r__data += value;);
#endif
  }
}

void THTensor_(bitand)(THTensor *r_, THTensor *t, real value)
{
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_HALF)
  (void)r_;
  (void)t;
  (void)value;
  return THError("bitand is only supported for integer type tensors");
#else
  THTensor_(resizeAs)(r_, t);
  int64_t r_Size = THTensor_(nElement)(r_);
  int r_Contig = THTensor_(isContiguous)(r_);
  int serial_path = 0;
  int tContig = THTensor_(isContiguous)(t);
  if (r_Contig && tContig) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    int64_t i;
    #pragma omp parallel for if(r_Size > TH_OMP_OVERHEAD_THRESHOLD * 100) private(i)
    for (i=0; i<r_Size; i++) {
      rp[i] = tp[i] & value;
    }
  } else {
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP) {
      serial_path = 1;
    } else {
      TH_TENSOR_APPLY2_OMP(r_Size, r_Contig, tContig, real, r_, real, t, *r__data = *t_data & value;, UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD);
    }
#else
    serial_path = 1;
#endif
  }
  if (serial_path) {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data & value;);
  }
#endif
}

#endif /* TH_GENERIC_FILE */
