#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiMarginCriterion.c"
#else

// TODO: improve error messages
void THNN_(MultiMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          int64_t reduction,
          int p,
          THTensor *weights,
          accreal margin_)
{
  real margin = TH_CONVERT_ACCREAL_TO_REAL(margin_);
  real *input_data, *weights_data;
  THIndex_t *target_data;
  int64_t nframe, dim;
  int64_t t, d;
  real sum;

  AT_CHECK(!input->is_empty() && input->dim() <= 2,
           "non-empty vector or matrix expected, got size: ", input->sizes());

  if (input->dim() <= 1)
  {
    nframe = 1;
    dim = THTensor_sizeLegacyNoScalars(input, 0);
  }
  else
  {
    nframe = input->size(0);
    dim = input->size(1);
    AT_CHECK(!target->is_empty() && (THTensor_nDimensionLegacyNoScalars(target) == 1) && (THTensor_sizeLegacyNoScalars(target, 0) == nframe),
             "inconsistent target size, got: ", target->sizes());
  }

  for (t = 0; t < nframe; t++)
  {
    THIndex_t idx = THIndexTensor_(get1d)(target, t);
    THArgCheck((idx >= TH_INDEX_BASE) && (idx < dim + TH_INDEX_BASE), 3,
	       "target out of range");
  }

  input = THTensor_(newContiguous)(input);
  target = THIndexTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;
  input_data = THTensor_(data)(input);
  target_data = THIndexTensor_(data)(target);
  weights_data = weights ? THTensor_(data)(weights) : NULL;

  if (reduction == Reduction::None)
  {
    THTensor_(resize1d)(output, nframe);

    for (t = 0; t < nframe; t++)
    {
      sum = 0;
      THIndex_t target_idx = target_data[t] - TH_INDEX_BASE;
      real input_target = input_data[target_idx];
      for (d = 0; d < dim; d++)
      {
        real z = margin - input_target + input_data[d];
        if (d == target_idx)
          continue;

        if (z > 0) {
          real h = (p==1) ? z : z*z;
          if(weights_data)
            h *= weights_data[target_idx];
          sum += h;
        }
      }

      sum /= dim;
      THTensor_(fastSet1d)(output, t, sum);
      input_data += dim;
    }
  }
  else
  {
    THTensor_(resize1d)(output, 1);

    sum = 0;
    for (t = 0; t < nframe; t++)
    {
      THIndex_t target_idx = target_data[t] - TH_INDEX_BASE;
      real input_target = input_data[target_idx];
      for (d = 0; d < dim; d++)
      {
        real z = margin - input_target + input_data[d];
        if (d == target_idx)
          continue;

        if (z > 0) {
          real h = (p==1) ? z : z*z;
          if(weights_data)
            h *= weights_data[target_idx];
          sum += h;
        }
      }
      input_data += dim;
    }

    sum /= dim;
    if(reduction == Reduction::ElementwiseMean)
      sum /= nframe;

    THTensor_(set1d)(output, 0, sum);
  }

  THTensor_(free)(input);
  THIndexTensor_(free)(target);
  if(weights)
    THTensor_(free)(weights);
}

void THNN_(MultiMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t reduction,
          int p,
          THTensor *weights,
          accreal margin_)
{
  real margin = TH_CONVERT_ACCREAL_TO_REAL(margin_);
  real *input_data;
  real *gradInput_data;
  THIndex_t *target_data;
  real *weights_data;
  int64_t nframe, dim;
  int64_t t, d;
  real g;

  AT_CHECK(!input->is_empty() && (input->dim() <= 2),
           "non-empty vector or matrix expected, got size: ", input->sizes());

  if (input->dim() <= 1)
  {
    nframe = 1;
    dim = THTensor_sizeLegacyNoScalars(input, 0);
  }
  else
  {
    nframe = input->size(0);
    dim = input->size(1);
    AT_CHECK(!target->is_empty() && (target->dim() <= 1) && (THTensor_sizeLegacyNoScalars(target, 0) == nframe),
             "inconsistent target size, got: ", target->sizes());
  }

  g = (reduction == Reduction::ElementwiseMean ? 1./((real)(nframe*dim)) : 1./((real)dim));

  input = THTensor_(newContiguous)(input);
  target = THIndexTensor_(newContiguous)(target);
  input_data = THTensor_(data)(input);

  THTensor_(resizeAs)(gradInput, input);
  THArgCheck(THTensor_(isContiguous)(gradInput), 5, "gradInput must be contiguous");
  gradInput_data = THTensor_(data)(gradInput);

  target_data = THIndexTensor_(data)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;
  weights_data = weights ? THTensor_(data)(weights) : NULL;

  for (t = 0; t < nframe; t++)
  {
    THIndex_t target_idx = target_data[t] - TH_INDEX_BASE;
    real input_target = input_data[target_idx];
    real gradInput_target = 0;
    for (d = 0; d < dim; d++)
    {
      real z = margin - input_target + input_data[d];
      if (d == target_idx)
        continue;

      if (z > 0)
      {
        real h = (p == 1) ? g : 2*g*z;
        if(weights_data)
          h *= weights_data[target_idx];
        gradInput_target -= h;
        gradInput_data[d] = h;
      }
      else
        gradInput_data[d] = 0;
    }
    gradInput_data[target_idx] = gradInput_target;

    input_data += dim;
    gradInput_data += dim;
  }
  gradInput_data = THTensor_(data)(gradInput);

  if (reduction != Reduction::None)
  {
    THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);
    for (t = 0; t < nframe * dim; t++) {
      gradInput_data[t] *= THTensor_(fastGetLegacy1dNoScalars)(gradOutput, 0);
    }
  }
  else
  {
    THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, nframe);
    for (t = 0; t < nframe; t++)
    {
      for (d = 0; d < dim; d++)
      {
        gradInput_data[t * dim + d] *= THTensor_(fastGetLegacy1dNoScalars)(gradOutput, t);
      }
    }
  }

  THTensor_(free)(input);
  THIndexTensor_(free)(target);
  if(weights)
    THTensor_(free)(weights);
}

#endif
