#ifndef CAFFE2_OPERATORS_MINMAX_OPS_H_
#define CAFFE2_OPERATORS_MINMAX_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class MaxMinOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MaxMinOpBase)

  bool RunOnDevice() override {
	std::vector<int> input0_dims(Input(0).dims().begin(), Input(0).dims().end());
	std::vector<int> output_dims(input0_dims);
    auto& input0 = Input(0);
	auto* output = Output(0);


	for (int i = 1; i < InputSize(); i++){
		std::vector<int> inputi_dims(Input(i).dims().begin(), Input(i).dims().end());
		output_dims = elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
				output_dims,
				inputi_dims);
	}

    output->Resize(output_dims);
	math::Broadcast<caffe2::Tensor<Context>, Context>(
		input0_dims.size(),
		input0_dims.data(),
		output_dims.size(),
		output_dims.data(),
		&input0,
		output,
		&context_);
	printf("InputSize is %d\n", InputSize());
    if (InputSize() == 1) {
      return true;
    }
    return this->Compute();
  }

  virtual bool Compute() = 0;
};

template <typename T, class Context>
class MaxOp : public MaxMinOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MaxOp(const OperatorDef& operator_def, Workspace* ws)
      : MaxMinOpBase<T, Context>(operator_def, ws) {}
  virtual ~MaxOp() noexcept {}
  bool Compute() override;
};

template <typename T, class Context>
class SelectGradientOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SelectGradientOpBase)

  bool RunOnDevice() override;
};

template <typename T, class Context>
class MaxGradientOp : public SelectGradientOpBase<T, Context> {
 public:
  MaxGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : SelectGradientOpBase<T, Context>(operator_def, ws) {}
  virtual ~MaxGradientOp() noexcept {}
};

template <typename T, class Context>
class MinOp : public MaxMinOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MinOp(const OperatorDef& operator_def, Workspace* ws)
      : MaxMinOpBase<T, Context>(operator_def, ws) {}
  virtual ~MinOp() noexcept {}
  bool Compute() override;
};

template <typename T, class Context>
class MinGradientOp : public SelectGradientOpBase<T, Context> {
 public:
  MinGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : SelectGradientOpBase<T, Context>(operator_def, ws) {}
  virtual ~MinGradientOp() noexcept {}
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MINMAX_OPS_H_
