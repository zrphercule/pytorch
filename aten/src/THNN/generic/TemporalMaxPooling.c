#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalMaxPooling.c"
#else

static inline void THNN_(TemporalMaxPooling_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         THIndexTensor *indices,
                         int kW,
                         int dW) {
  int64_t niframe;
  int64_t framesize;
  int64_t noframe;

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  int ndims = input->dim();

  if (input->dim() == 3)
  {
    dimS = 1;
    dimF = 2;
  }

  niframe = input->size(dimS);
  framesize = input->size(dimF);
  noframe = (niframe - kW) / dW + 1;

  THArgCheck(kW > 0, 5,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(dW > 0, 6,
             "stride should be greater than zero, but got dW: %d", dW);

  THNN_ARGCHECK(!input->is_empty() && (input->dim() == 2 || input->dim() == 3), 2, input,
                "non-empty 2D or 3D (batch mode) tensor expected for input, but got: %s");
  THArgCheck(input->size(dimS) >= kW, 2,
             "input sequence smaller than kernel size. Got: %d, Expected: %d",
             input->size(dimS), kW);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndims, dimS, noframe);
    THNN_CHECK_DIM_SIZE(gradOutput, ndims, dimF, framesize)
  }
  if (indices != NULL) {
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndims, dimS, noframe);
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndims, dimF, framesize);
  }
}

void THNN_(TemporalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kW,
          int dW)
{
  int64_t niframe;
  int64_t framesize;
  int64_t noframe;

  scalar_t *input_data;
  scalar_t *output_data;
  THIndex_t *indices_data;

  int64_t t, y;

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  THNN_(TemporalMaxPooling_shapeCheck)(state, input, NULL, NULL, kW, dW);

  if (input->dim() == 3)
  {
    dimS = 1;
    dimF = 2;
  }

  /* sizes */
  niframe = input->size(dimS);
  framesize = input->size(dimF);
  noframe = (niframe - kW) / dW + 1;

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (input->dim() == 2)
  {
    /* resize output */
    THTensor_(resize2d)(output, noframe, framesize);

    /* indices will contain index locations for each output point */
    THIndexTensor_(resize2d)(indices, noframe, framesize);

    /* get raw pointers */
    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();
    indices_data = THIndexTensor_(data)(indices);

    for(t = 0; t < noframe; t++)
    {
      scalar_t *ip = input_data + t*framesize*dW;
      scalar_t *op = output_data + t*framesize;
      THIndex_t *xp = indices_data + t*framesize;
#pragma omp parallel for private(y)
      for(y = 0; y < framesize; y++)
      {
        /* compute local max: */
        int64_t maxindex = -1;
        scalar_t maxval = -THInf;
        int64_t x;
        for(x = 0; x < kW; x++)
        {
          scalar_t val = ip[x*framesize+y];
          if (val > maxval)
          {
            maxval = val;
            maxindex = x;
          }
        }

        /* set output to local max */
        op[y] = maxval;
        xp[y] = (scalar_t)maxindex;
      }
    }
  }
  else
  {
    /* number of batch frames */
    int64_t nbframe = input->size(0);
    int64_t i;

    /* resize output */
    THTensor_(resize3d)(output, nbframe, noframe, framesize);

    /* indices will contain index locations for each output point */
    THIndexTensor_(resize3d)(indices, nbframe, noframe, framesize);

    /* get raw pointers */
    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();
    indices_data = THIndexTensor_(data)(indices);

    for(i = 0; i < nbframe; i++)
    {
      scalar_t *inputSample_data = input_data + i*niframe*framesize;
      scalar_t *outputSample_data = output_data + i*noframe*framesize;
      THIndex_t *indicesSample_data = indices_data + i*noframe*framesize;

      for(t = 0; t < noframe; t++)
      {
        scalar_t *ip = inputSample_data + t*framesize*dW;
        scalar_t *op = outputSample_data + t*framesize;
        THIndex_t *xp = indicesSample_data + t*framesize;

#pragma omp parallel for private(y)
        for(y = 0; y < framesize; y++)
        {
          /* compute local max: */
          int64_t maxindex = -1;
          scalar_t maxval = -THInf;
          int64_t x;
          for(x = 0; x < kW; x++)
          {
            scalar_t val = ip[x*framesize+y];
            if (val > maxval)
            {
              maxval = val;
              maxindex = x;
            }
          }

          /* set output to local max */
          op[y] = maxval;
          xp[y] = (scalar_t)maxindex;
        }
      }
    }
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(input);

}

void THNN_(TemporalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kW,
          int dW)
{
  int64_t niframe;
  int noframe;
  int64_t framesize;

  scalar_t *gradInput_data;
  scalar_t *gradOutput_data;
  THIndex_t *indices_data;

  int64_t t, y;

  THNN_(TemporalMaxPooling_shapeCheck)(state, input, gradOutput, indices, kW, dW);
  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize and zero */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  if (input->dim() == 3)
  {
    dimS = 1;
    dimF = 2;
  }
  /* sizes */
  niframe = input->size(dimS);
  noframe = gradOutput->size(dimS);
  framesize = gradOutput->size(dimF);

  /* get raw pointers */
  gradInput_data = gradInput->data<scalar_t>();
  gradOutput_data = gradOutput->data<scalar_t>();
  indices_data = THIndexTensor_(data)(indices);

  if (input->dim() == 2)
  {
    for(t = 0; t < noframe; t++)
    {
      scalar_t *gip = gradInput_data + t*framesize*dW;
      scalar_t *gop = gradOutput_data + t*framesize;
      THIndex_t *xp = indices_data + t*framesize;
#pragma omp parallel for private(y)
      for(y = 0; y < framesize; y++)
      {
        /* compute local max: */
        int64_t maxindex = (int64_t)xp[y];
	if (maxindex != -1)
	  gip[maxindex*framesize+y] += gop[y];
      }
    }
  }
  else
  {
    /* number of batch frames */
    int64_t nbframe = input->size(0);
    int64_t i;

    for(i = 0; i < nbframe; i++)
    {
      scalar_t *gradInputSample_data = gradInput_data + i*niframe*framesize;
      scalar_t *gradOutputSample_data = gradOutput_data + i*noframe*framesize;
      THIndex_t *indicesSample_data = indices_data + i*noframe*framesize;

      for(t = 0; t < noframe; t++)
      {
        scalar_t *gip = gradInputSample_data + t*framesize*dW;
        scalar_t *gop = gradOutputSample_data + t*framesize;
        THIndex_t *xp = indicesSample_data + t*framesize;
#pragma omp parallel for private(y)
        for(y = 0; y < framesize; y++)
        {
          /* compute local max: */
          int64_t maxindex = (int64_t)xp[y];
	  if (maxindex != -1)
	    gip[maxindex*framesize+y] += gop[y];
        }
      }
    }
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
