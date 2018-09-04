#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionMap.c"
#else

void THNN_(SpatialConvolutionMap_updateOutput)(
  THNNState *state, THTensor *input, THTensor *output, THTensor *weight, THTensor *bias,
  THTensor *connTable, int nInputPlane, int nOutputPlane,
  int dW, int dH)
{
  THArgCheck(
    weight != NULL && !weight->is_empty() && weight->dim() == 3
    && connTable != NULL && connTable->size(0) == weight->size(0), 4,
    "non-empty 3D weight tensor expected (connTable:size(%d) x kH x kW)", TH_INDEX_BASE
  );

  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  int64_t nbatch = 1;

  THArgCheck(!input->is_empty() && (input->dim() == 3 || input->dim() == 4), 2, "non-empty 3D or 4D(batch mode) tensor expected");

  if (input->dim() == 4)
  {
    nbatch = input->size(0);
    dimc++;
    dimw++;
    dimh++;
  }

  const int64_t kH       = weight->size(1);
  const int64_t kW       = weight->size(2);

  THArgCheck(input->size(dimc) >= nInputPlane, 2, "invalid number of input planes");
  THArgCheck(input->size(dimw) >= kW && input->size(dimh) >= kH, 2, "input image smaller than kernel size");

  const int64_t input_w  = input->size(dimw);
  const int64_t input_h  = input->size(dimh);
  const int64_t output_w = (input_w - kW) / dW + 1;
  const int64_t output_h = (input_h - kH) / dH + 1;

  if (input->dim() == 3)
    THTensor_(resize3d)(output, nOutputPlane, output_h, output_w);
  else
    THTensor_(resize4d)(output, input->size(0), nOutputPlane, output_h, output_w);

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  output = THTensor_(newContiguous)(output);
  weight = THTensor_(newContiguous)(weight);
  bias = bias ? THTensor_(newContiguous)(bias) : bias;
  connTable = THTensor_(newContiguous)(connTable);

  /* get raw pointers */
  scalar_t *input_data = input->data<scalar_t>();
  scalar_t *output_data = output->data<scalar_t>();
  scalar_t *weight_data = weight->data<scalar_t>();
  scalar_t *bias_data = bias->data<scalar_t>();
  scalar_t *connTable_data = connTable->data<scalar_t>();

  int64_t p;
#pragma omp parallel for private(p)
  for (p = 0; p < nOutputPlane; p++)
  {
    int64_t m;
    for (m = 0; m < nbatch; m++)
    {
      /* add bias */
      scalar_t *ptr_output = output_data + p*output_w*output_h + m*nOutputPlane*output_w*output_h;
      int64_t j, k;
      scalar_t z= bias_data[p];
      for (j = 0; j < output_h*output_w; j++)
        ptr_output[j] = z;

      /* convolve all maps */
      int nweight = connTable->size(0);
      for (k = 0; k < nweight; k++)
      {
        /* get offsets for input/output */
        int o = (int)connTable_data[k*2+1] - TH_INDEX_BASE;
        int i = (int)connTable_data[k*2+0] - TH_INDEX_BASE;

        if (o == p)
        {
          THTensor_(validXCorr2Dptr)(
            output_data + o*output_w*output_h + m*nOutputPlane*output_w*output_h,
            1.0,
            input_data + i*input_w*input_h + m*nInputPlane*input_w*input_h, input_h, input_w,
            weight_data + k*kW*kH,
            kH, kW,
            dH, dW
          );
        }
      }
    }
  }

  /* clean up */
  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(output);
  c10::raw::intrusive_ptr::decref(weight);
  if (bias) c10::raw::intrusive_ptr::decref(bias);
  c10::raw::intrusive_ptr::decref(connTable);
}

void THNN_(SpatialConvolutionMap_updateGradInput)(
  THNNState *state, THTensor *input, THTensor *gradOutput, THTensor *gradInput, THTensor *weight, THTensor *bias,
  THTensor *connTable, int nInputPlane, int nOutputPlane,
  int dW, int dH)
{
  THArgCheck(
    weight != NULL && !weight->is_empty() && weight->dim() == 3
    && connTable != NULL && connTable->size(0) == weight->size(0), 5,
    "non-empty 3D weight tensor expected (connTable:size(%d) x kH x kW)", TH_INDEX_BASE
  );

  /* and dims */
  int dimw = 2;
  int dimh = 1;
  int64_t nbatch = 1;
  if (input->dim() == 4)
  {
    nbatch = input->size(0);
    dimw++;
    dimh++;
  }

  const int64_t input_h  = input->size(dimh);
  const int64_t input_w  = input->size(dimw);
  const int64_t output_h = gradOutput->size(dimh);
  const int64_t output_w = gradOutput->size(dimw);
  const int64_t kH       = weight->size(1);
  const int64_t kW       = weight->size(2);

  /* contiguous */
  gradInput = THTensor_(newContiguous)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  weight = THTensor_(newContiguous)(weight);
  connTable = THTensor_(newContiguous)(connTable);

  /* Resize/Zero */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* get raw pointers */
  scalar_t *gradInput_data = gradInput->data<scalar_t>();
  scalar_t *gradOutput_data = gradOutput->data<scalar_t>();
  scalar_t *weight_data = weight->data<scalar_t>();
  scalar_t *connTable_data = connTable->data<scalar_t>();

  int64_t p;
#pragma omp parallel for private(p)
  for (p = 0; p < nInputPlane; p++)
  {
    int64_t m;
    for (m = 0; m < nbatch; m++)
    {
      int64_t k;
      /* backward all */
      int nkernel = connTable->size(0);
      for (k = 0; k < nkernel; k++)
      {
        int o = (int)connTable_data[k*2+1] - TH_INDEX_BASE;
        int i = (int)connTable_data[k*2+0] - TH_INDEX_BASE;
        if (i == p)
        {
          /* gradient to input */
          THTensor_(fullConv2Dptr)(
            gradInput_data + i*input_w*input_h + m*nInputPlane*input_w*input_h, 1.0,
            gradOutput_data + o*output_w*output_h + m*nOutputPlane*output_w*output_h,  output_h,  output_w,
            weight_data + k*kW*kH, kH, kW, dH, dW
          );
        }
      }
    }
  }

  /* clean up */
  c10::raw::intrusive_ptr::decref(gradInput);
  c10::raw::intrusive_ptr::decref(gradOutput);
  c10::raw::intrusive_ptr::decref(weight);
  c10::raw::intrusive_ptr::decref(connTable);
}

void THNN_(SpatialConvolutionMap_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *connTable,
          int nInputPlane,
          int nOutputPlane,
          int dW, int dH,
          accreal scale_)
{
  scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THArgCheck(
    gradWeight != NULL && !gradWeight->is_empty() && gradWeight->dim() == 3
    && connTable != NULL && connTable->size(0) == gradWeight->size(0), 5,
    "3D gradWeight tensor expected (connTable:size(%d) x kH x kW)", TH_INDEX_BASE
  );

  /* and dims */
  int dimw = 2;
  int dimh = 1;
  int64_t nbatch = 1;
  if (input->dim() == 4)
  {
    nbatch = input->size(0);
    dimw++;
    dimh++;
  }

  const int64_t input_h  = input->size(dimh);
  const int64_t input_w  = input->size(dimw);
  const int64_t output_h = gradOutput->size(dimh);
  const int64_t output_w = gradOutput->size(dimw);
  const int64_t kH       = gradWeight->size(1);
  const int64_t kW       = gradWeight->size(2);

  /* contiguous */
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  THArgCheck(THTensor_(isContiguous)(gradWeight), 4, "gradWeight needs to be contiguous");
  THArgCheck(THTensor_(isContiguous)(gradBias), 5, "gradBias needs to be contiguous");

  /* get raw pointers */
  scalar_t *input_data = input->data<scalar_t>();
  scalar_t *gradOutput_data = gradOutput->data<scalar_t>();
  scalar_t *gradWeight_data = gradWeight->data<scalar_t>();
  scalar_t *gradBias_data = gradBias->data<scalar_t>();


  int64_t k;
  /* gradients wrt bias */
#pragma omp parallel for private(k)
  for (k = 0; k < nOutputPlane; k++)
  {
    int64_t m;
    for (m = 0; m < nbatch; m++)
    {
      scalar_t *ptr_gradOutput = gradOutput_data + k*output_w*output_h + m*nOutputPlane*output_w*output_h;
      int64_t l;
      for (l = 0; l < output_h*output_w; l++)
        gradBias_data[k] += scale*ptr_gradOutput[l];
    }
  }

  /* gradients wrt weight */
  const int nkernel = connTable->size(0);
#pragma omp parallel for private(k)
  for (k = 0; k < nkernel; k++)
  {
    int64_t m;
    for (m = 0; m < nbatch; m++)
    {
      int o = (int)THTensor_(get2d)(connTable,k,1) - TH_INDEX_BASE;
      int i = (int)THTensor_(get2d)(connTable,k,0) - TH_INDEX_BASE;

      /* gradient to kernel */
      THTensor_(validXCorr2DRevptr)(
        gradWeight_data + k*kW*kH,
        scale,
        input_data + i*input_w*input_h + m*nInputPlane*input_w*input_h, input_h, input_w,
        gradOutput_data + o*output_w*output_h + m*nOutputPlane*output_w*output_h , output_h, output_w,
        dH, dW
      );
    }
  }

  /* clean up */
  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
