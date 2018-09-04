#include "caffe2/operators/top_k.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
struct ValueComp {
  bool operator()(
      const std::pair<T, TIndex>& lhs,
      const std::pair<T, TIndex>& rhs) const {
    return lhs.first > rhs.first ||
        (lhs.first == rhs.first && lhs.second < rhs.second);
  }
};

template <typename T>
void GetTopK(
    const T* input,
    const TIndex n,
    const TIndex k,
    const TIndex src_offset,
    const TIndex dst_offset,
    const TIndex stride,
    T* values,
    TIndex* indices,
    TIndex* flatten_indices) {
  const T* src_ptr = input + src_offset;
  std::vector<std::pair<T, TIndex>> heap_data;
  heap_data.reserve(k);
  for (TIndex i = 0; i < k && i < n; ++i) {
    heap_data.emplace_back(*src_ptr, i);
    src_ptr += stride;
  }
  std::priority_queue<
      std::pair<T, TIndex>,
      std::vector<std::pair<T, TIndex>>,
      ValueComp<T>>
      pq(ValueComp<T>(), std::move(heap_data));
  for (TIndex i = k; i < n; ++i) {
    if (pq.top().first < *src_ptr) {
      pq.pop();
      pq.emplace(*src_ptr, i);
    }
    src_ptr += stride;
  }
  TIndex dst_pos = dst_offset + (std::min(k, n) - 1) * stride;
  while (!pq.empty()) {
    const auto& item = pq.top();
    values[dst_pos] = item.first;
    indices[dst_pos] = item.second;
    if (flatten_indices != nullptr) {
      flatten_indices[dst_pos] = src_offset + item.second * stride;
    }
    pq.pop();
    dst_pos -= stride;
  }
}

template <typename T>
void SetTopKGradient(
    const T* values,
    const TIndex* indices,
    const int k,
    const TIndex src_offset,
    const TIndex dst_offset,
    const TIndex stride,
    T* gradient) {
  TIndex src_pos = src_offset;
  for (int i = 0; i < k; ++i) {
    if (indices[src_pos] < 0) {
      continue;
    }
    gradient[dst_offset + indices[src_pos] * stride] = values[src_pos];
    src_pos += stride;
  }
}

} // namespace

template <typename T, class Context>
bool TopKOp<T, Context>::RunOnDevice() {
  const auto& input = Input(0);
  auto* values = Output(0);
  auto* indices = Output(1);
  auto* flatten_indices = OutputSize() > 2 ? Output(2) : nullptr;

  const std::vector<TIndex>& input_dims = input.dims();
  if (axis_ == -1) {
    axis_ = input_dims.size() - 1;
  }
  CAFFE_ENFORCE_GE(axis_, 0);
  CAFFE_ENFORCE_LT(axis_, input_dims.size());

  std::vector<TIndex> output_dims = input_dims;
  output_dims[axis_] = k_;
  values->Resize(output_dims);
  indices->Resize(output_dims);
  if (flatten_indices != nullptr) {
    flatten_indices->Resize(indices->size());
  }
  const T* input_data = input.template data<T>();
  T* values_data = values->template mutable_data<T>();
  TIndex* indices_data = indices->template mutable_data<TIndex>();
  TIndex* flatten_indices_data = flatten_indices == nullptr
      ? nullptr
      : flatten_indices->template mutable_data<TIndex>();
  // init values as the default value
  math::Set<T, Context>(values->size(), T(0), values_data, &context_);
  math::Set<TIndex, Context>(
      indices->size(), TIndex(-1), indices_data, &context_);
  if (flatten_indices_data != nullptr) {
    math::Set<TIndex, Context>(
        flatten_indices->size(), TIndex(-1), flatten_indices_data, &context_);
  }

  const TIndex prev_size = std::accumulate(
      input_dims.cbegin(),
      input_dims.cbegin() + axis_,
      TIndex(1),
      std::multiplies<TIndex>());
  const TIndex next_size = std::accumulate(
      input_dims.cbegin() + axis_ + 1,
      input_dims.cend(),
      TIndex(1),
      std::multiplies<TIndex>());
  const TIndex src_offset_stride = input_dims[axis_] * next_size;
  const TIndex dst_offset_stride = k_ * next_size;
  TIndex src_offset = 0;
  TIndex dst_offset = 0;
  for (TIndex i = 0; i < prev_size; ++i) {
    for (TIndex j = 0; j < next_size; ++j) {
      GetTopK(
          input_data,
          input_dims[axis_],
          k_,
          src_offset + j,
          dst_offset + j,
          next_size,
          values_data,
          indices_data,
          flatten_indices_data);
    }
    src_offset += src_offset_stride;
    dst_offset += dst_offset_stride;
  }
  return true;
}

template <typename T, class Context>
bool TopKGradientOp<T, Context>::RunOnDevice() {
  const auto& values = Input(0);
  const auto& indices = Input(1);
  const auto& original_input = Input(2);
  auto* output = Output(0);
  const std::vector<TIndex>& values_dims = values.dims();
  const std::vector<TIndex>& origin_dims = original_input.dims();
  CAFFE_ENFORCE_EQ(values_dims.size(), origin_dims.size());
  output->Resize(origin_dims);
  const T* values_data = values.template data<T>();
  const TIndex* indices_data = indices.template data<TIndex>();
  T* output_data = output->template mutable_data<T>();
  if (axis_ == -1) {
    axis_ = values_dims.size() - 1;
  }
  const int k = values_dims[axis_];
  math::Set<T, Context>(output->size(), T(0), output_data, &context_);
  const TIndex prev_size = std::accumulate(
      values_dims.cbegin(),
      values_dims.cbegin() + axis_,
      TIndex(1),
      std::multiplies<TIndex>());
  const TIndex next_size = std::accumulate(
      values_dims.cbegin() + axis_ + 1,
      values_dims.cend(),
      TIndex(1),
      std::multiplies<TIndex>());
  const TIndex src_offset_stride = k * next_size;
  const TIndex dst_offset_stride = origin_dims[axis_] * next_size;
  TIndex src_offset = 0;
  TIndex dst_offset = 0;
  for (TIndex i = 0; i < prev_size; ++i) {
    for (TIndex j = 0; j < next_size; ++j) {
      SetTopKGradient(
          values_data,
          indices_data,
          k,
          src_offset + j,
          dst_offset + j,
          next_size,
          output_data);
    }
    src_offset += src_offset_stride;
    dst_offset += dst_offset_stride;
  }
  return true;
}

REGISTER_CPU_OPERATOR(TopK, TopKOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(TopKGradient, TopKGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(TopK)
    .NumInputs(1)
    .NumOutputs(2, 3)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out = {in[0], in[0]};
      ArgumentHelper helper(def);
      auto k = helper.GetSingleArgument("k", -1);
      auto dims_size = in[0].dims_size();
      out[0].set_dims(dims_size - 1, k);
      out[1].set_dims(dims_size - 1, k);
      out[1].set_data_type(TensorProto_DataType_INT32);
      if (def.output_size() > 2) {
        TensorShape flatten_indices_shape;
        flatten_indices_shape.set_data_type(TensorProto_DataType_INT32);
        flatten_indices_shape.add_dims(
            std::accumulate(
                in[0].dims().begin(),
                in[0].dims().end() - 1,
                1,
                std::multiplies<long>()) *
            k);
        out.push_back(flatten_indices_shape);
      }
      return out;
    })
    .SetDoc(R"DOC(
Retrieve the top-K elements of the last dimension. Given an input tensor of shape $(a_1, a_2, ..., a_n, r)$ and integer argument `k`, return up to three outputs:

1. Value tensor of shape $(a_1, a_2, ..., a_n, k)$ which contains the values of the top k elements along the last dimension
2. Index tensor of shape $(a_1, a_2, ..., a_n, k)$ which contains the indices of the top k elements (original indices from the input tensor).
3. [OPTIONAL] Flattened index tensor of shape $(a_1 * a_2 * ... * a_n * k,)$.

Given two equivalent values, this operator uses the indices along the last dimension as a tiebreaker. That is, the element with the lower index will appear first.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "TopK",
    ["X"],
    ["Values", "Indices", "Flattened_indices"],
    k=2
)

workspace.FeedBlob("X", np.random.randint(10, size=(3,3,3)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Values:", workspace.FetchBlob("Values"))
print("Indices:", workspace.FetchBlob("Indices"))
print("Flattened_indices:", workspace.FetchBlob("Flattened_indices"))

```

**Result**

```

X:
[[[6. 7. 0.]
  [8. 7. 7.]
  [1. 5. 6.]]

 [[0. 6. 1.]
  [2. 8. 4.]
  [1. 2. 9.]]

 [[4. 3. 7.]
  [0. 1. 7.]
  [0. 1. 8.]]]
Values:
[[[7. 6.]
  [8. 7.]
  [6. 5.]]

 [[6. 1.]
  [8. 4.]
  [9. 2.]]

 [[7. 4.]
  [7. 1.]
  [8. 1.]]]
Indices:
[[[1 0]
  [0 1]
  [2 1]]

 [[1 2]
  [1 2]
  [2 1]]

 [[2 0]
  [2 1]
  [2 1]]]
Flattened_indices: [ 1  0  3  4  8  7 10 11 13 14 17 16 20 18 23 22 26 25]

```

</details>

  )DOC")
    .Input(
      0,
      "X",
      "(*Tensor`<float>`*): input tensor of shape $(a_1, a_2, ..., a_n, r)$")
    .Output(
        0,
        "Values",
        "(*Tensor`<float>`*): output tensor of shape $(a_1, a_2, ..., a_n, k)$")
    .Output(
        1,
        "Indices",
        "(*Tensor`<int>`*): tensor of indices of shape $(a_1, a_2, ..., a_n, k)$; indices values refer to each element's index in the last dimension of the `X` input tensor")
    .Output(
        2,
        "Flattened_indices",
        "(*Tensor`<int>`*): tensor of indices of shape $(a_1 * a_2 * ... * a_n * k,)$; indices values refer to each element's index in the flattened input tensor `X`")
    .Arg("k", "(*int*): number of top elements to retrieve");

OPERATOR_SCHEMA(TopKGradient).NumInputs(3).NumOutputs(1);

class GetTopKGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TopKGradient",
        "",
        vector<string>{GO(0), O(1), I(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(TopK, GetTopKGradient);

} // namespace caffe2
