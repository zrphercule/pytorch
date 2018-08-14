#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/ELU.cu"
#else

#include "../common.h"


void THNN_(ELU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal alpha,
           accreal scale,
           accreal input_scale,
           bool inplace)
{
  real negcoef = ScalarConvert<accreal, real>::to(alpha * scale);
  real poscoef = ScalarConvert<accreal, real>::to(scale * input_scale);
  real negiptcoef = ScalarConvert<accreal, real>::to(input_scale);
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1<real>(state, input, ELUupdateOutputIP_functor<real>(negcoef, poscoef, negiptcoef));
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2<real, real>(state, output, input, ELUupdateOutput_functor<real>(negcoef, poscoef, negiptcoef));
  }
}


void THNN_(ELU_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           accreal alpha,
           accreal scale,
           accreal input_scale)
{
  real negcoef = ScalarConvert<accreal, real>::to(alpha * scale);
  real poscoef = ScalarConvert<accreal, real>::to(scale * input_scale);
  real negiptcoef = ScalarConvert<accreal, real>::to(input_scale);
  THCUNN_check_nElement(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3<real, real, real>(state, gradInput, output, gradOutput, ELUupdateGradInput_functor<real>(negcoef, poscoef, negiptcoef));
}

#endif
