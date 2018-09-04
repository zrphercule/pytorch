#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalConvolution.c"
#else

static inline void THNN_(TemporalConvolution_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         int kW,
                         int dW,
                         int *inputFrameSize) {

  THArgCheck(kW > 0, 9,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(dW > 0, 11,
             "stride should be greater than zero, but got dW: %d", dW);

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  if (input->dim() == 3)
  {
    dimS = 1;
    dimF = 2;
  }
  THNN_ARGCHECK(!input->is_empty() && (input->dim() == 2 || input->dim() == 3), 2, input,
                  "non-empty 2D or 3D (batch mode) tensor expected for input, but got: %s");
  if (inputFrameSize != NULL) {
    THArgCheck(input->size(dimF) == *inputFrameSize, 2,
               "invalid input frame size. Got: %d, Expected: %d",
               input->size(dimF), *inputFrameSize);
  }
  THArgCheck(input->size(dimS) >= kW, 2,
             "input sequence smaller than kernel size. Got: %d, Expected: %d",
             input->size(dimS), kW);
}

void THNN_(TemporalConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW,
          int dW,
          int inputFrameSize,
          int outputFrameSize)
{
  THTensor *outputWindow, *inputWindow;
  int nInputFrame, nOutputFrame;
  int64_t k, i;

  int dimS = 0; // sequence dimension

  if (input->dim() == 3)
  {
    dimS = 1;
  }

  THArgCheck(THTensor_(isContiguous)(weight), 4, "weight must be contiguous");
  THArgCheck(!bias || THTensor_(isContiguous)(bias), 5, "bias must be contiguous");
  THNN_(TemporalConvolution_shapeCheck)
       (state, input, kW, dW, &inputFrameSize);
  input = THTensor_(newContiguous)(input);
  outputWindow = THTensor_(new)();
  inputWindow = THTensor_(new)();

  nInputFrame = input->size(dimS);
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  if (input->dim() == 2)
  {
    THTensor_(resize2d)(output,
                        nOutputFrame,
                        outputFrameSize);

    /* bias first */
    for(k = 0; k < nOutputFrame; k++)
    {
      THTensor_(select)(outputWindow, output, 0, k);
      THTensor_(copy)(outputWindow, bias);
    }

    /* ouch */
    for(k = 0; nOutputFrame > 0; k++)
    {
      int64_t outputFrameStride = (kW-1)/dW+1;
      int64_t inputFrameStride = outputFrameStride*dW;
      int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THTensor_(setStorage2d)(inputWindow, THTensor_getStoragePtr(input),
                              input->storage_offset()+k*dW*input->size(1),
                              nFrame, inputFrameStride*input->size(1),
                              kW*input->size(1), 1);

      THTensor_(setStorage2d)(outputWindow, THTensor_getStoragePtr(output),
                              output->storage_offset() + k*output->size(1),
                              nFrame, outputFrameStride*output->size(1),
                              output->size(1), 1);

      THTensor *tweight = THTensor_(new)();
      THTensor_(transpose)(tweight, weight, 0, 1);
      THTensor_(addmm)(outputWindow, 1, outputWindow, 1, inputWindow, tweight);
      c10::raw::intrusive_ptr::decref(tweight);
    }
  }
  else
  {
    THTensor *outputSample = THTensor_(new)();
    THTensor *inputSample = THTensor_(new)();
    int nBatchFrame = input->size(0);

    THTensor_(resize3d)(output,
                        nBatchFrame,
                        nOutputFrame,
                        outputFrameSize);

    for(i = 0; i < nBatchFrame; i++)
    {
      THTensor_(select)(outputSample, output, 0, i);
      THTensor_(select)(inputSample, input, 0, i);
      int64_t nOutputSampleFrame = nOutputFrame;

      /* bias first */
      for(k = 0; k < nOutputFrame; k++)
      {
        THTensor_(select)(outputWindow, outputSample, 0, k);
        THTensor_(copy)(outputWindow, bias);
      }

      /* ouch */
      for(k = 0; nOutputSampleFrame > 0; k++)
      {
        int64_t outputFrameStride = (kW-1)/dW+1;
        int64_t inputFrameStride = outputFrameStride*dW;
        int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THTensor_(setStorage2d)(inputWindow, THTensor_getStoragePtr(inputSample),
                                inputSample->storage_offset()+k*dW*inputSample->size(1),
                                nFrame, inputFrameStride*inputSample->size(1),
                                kW*inputSample->size(1), 1);

        THTensor_(setStorage2d)(outputWindow, THTensor_getStoragePtr(outputSample),
                                outputSample->storage_offset() + k*outputSample->size(1),
                                nFrame, outputFrameStride*outputSample->size(1),
                                outputSample->size(1), 1);

        THTensor *tweight = THTensor_(new)();
        THTensor_(transpose)(tweight, weight, 0, 1);
        THTensor_(addmm)(outputWindow, 1, outputWindow, 1, inputWindow, tweight);
        c10::raw::intrusive_ptr::decref(tweight);
      }
    }
    c10::raw::intrusive_ptr::decref(outputSample);
    c10::raw::intrusive_ptr::decref(inputSample);
  }

  c10::raw::intrusive_ptr::decref(outputWindow);
  c10::raw::intrusive_ptr::decref(inputWindow);
  c10::raw::intrusive_ptr::decref(input);

}

void THNN_(TemporalConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW,
          int dW)
{
  int64_t nInputFrame;
  int64_t nOutputFrame;

  THTensor *gradOutputWindow;
  THTensor *gradInputWindow;
  int64_t k, i;

  int dimS = 0; // sequence dimension

  if (gradOutput->dim() == 3)
  {
    dimS = 1;
  }

  THArgCheck(THTensor_(isContiguous)(weight), 4, "weight must be contiguous");
  THNN_(TemporalConvolution_shapeCheck)(
        state, input, kW, dW, NULL);
  nInputFrame = input->size(dimS);
  nOutputFrame = gradOutput->size(dimS);

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  gradOutputWindow = THTensor_(new)();
  gradInputWindow = THTensor_(new)();

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (gradOutput->dim() == 2)
  {
    /* ouch */
    for(k = 0; nOutputFrame > 0; k++)
    {
      int64_t outputFrameStride = (kW-1)/dW+1;
      int64_t inputFrameStride = outputFrameStride*dW;
      int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THTensor_(setStorage2d)(gradOutputWindow, THTensor_getStoragePtr(gradOutput),
                              gradOutput->storage_offset() + k*gradOutput->size(1),
                              nFrame, outputFrameStride*gradOutput->size(1),
                              gradOutput->size(1), 1);

      THTensor_(setStorage2d)(gradInputWindow, THTensor_getStoragePtr(gradInput),
                              gradInput->storage_offset()+k*dW*gradInput->size(1),
                              nFrame, inputFrameStride*gradInput->size(1),
                              kW*gradInput->size(1), 1);

      THTensor_(addmm)(gradInputWindow, 1, gradInputWindow, 1, gradOutputWindow, weight);
    }
  }
  else
  {
    THTensor *gradOutputSample = THTensor_(new)();
    THTensor *gradInputSample = THTensor_(new)();
    int nBatchFrame = input->size(0);

    for(i = 0; i < nBatchFrame; i++)
    {
      THTensor_(select)(gradOutputSample, gradOutput, 0, i);
      THTensor_(select)(gradInputSample, gradInput, 0, i);
      int nOutputSampleFrame = nOutputFrame;

      /* ouch */
      for(k = 0; nOutputSampleFrame > 0; k++)
      {
        int64_t outputFrameStride = (kW-1)/dW+1;
        int64_t inputFrameStride = outputFrameStride*dW;
        int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THTensor_(setStorage2d)(gradOutputWindow, THTensor_getStoragePtr(gradOutputSample),
                                gradOutputSample->storage_offset() + k*gradOutputSample->size(1),
                                nFrame, outputFrameStride*gradOutputSample->size(1),
                                gradOutputSample->size(1), 1);

        THTensor_(setStorage2d)(gradInputWindow, THTensor_getStoragePtr(gradInputSample),
                                gradInputSample->storage_offset()+k*dW*gradInputSample->size(1),
                                nFrame, inputFrameStride*gradInputSample->size(1),
                                kW*gradInputSample->size(1), 1);

        THTensor_(addmm)(gradInputWindow, 1, gradInputWindow, 1, gradOutputWindow, weight);
      }
    }
    c10::raw::intrusive_ptr::decref(gradOutputSample);
    c10::raw::intrusive_ptr::decref(gradInputSample);
  }

  c10::raw::intrusive_ptr::decref(gradOutputWindow);
  c10::raw::intrusive_ptr::decref(gradInputWindow);
  c10::raw::intrusive_ptr::decref(gradOutput);
  c10::raw::intrusive_ptr::decref(input);

}

void THNN_(TemporalConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW,
          int dW,
          accreal scale_)
{
  scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  int64_t nInputFrame;
  int64_t nOutputFrame;

  THTensor *gradOutputWindow;
  THTensor *inputWindow;
  int64_t k, i;

  int dimS = 0; // sequence dimension

  if (gradOutput->dim() == 3)
  {
    dimS = 1;
  }

  THNN_(TemporalConvolution_shapeCheck)(
        state, input, kW, dW, NULL);
  nInputFrame = input->size(dimS);
  nOutputFrame = gradOutput->size(dimS);

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  gradOutputWindow = THTensor_(new)();
  inputWindow = THTensor_(new)();

  if (input->dim() == 2)
  {
    /* bias first */
    for(k = 0; k < nOutputFrame; k++)
    {
      THTensor_(select)(gradOutputWindow, gradOutput, 0, k);
      THTensor_(cadd)(gradBias, gradBias, scale, gradOutputWindow);
    }

    /* ouch */
    for(k = 0; nOutputFrame > 0; k++)
    {
      int64_t outputFrameStride = (kW-1)/dW+1;
      int64_t inputFrameStride = outputFrameStride*dW;
      int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THTensor_(setStorage2d)(inputWindow, THTensor_getStoragePtr(input),
                              input->storage_offset()+k*dW*input->size(1),
                              nFrame, inputFrameStride*input->size(1),
                              kW*input->size(1), 1);

      THTensor_(setStorage2d)(gradOutputWindow, THTensor_getStoragePtr(gradOutput),
                              gradOutput->storage_offset() + k*gradOutput->size(1),
                              nFrame, outputFrameStride*gradOutput->size(1),
                              gradOutput->size(1), 1);

      THTensor *tgradOutputWindow = THTensor_(new)();
      THTensor_(transpose)(tgradOutputWindow, gradOutputWindow, 0, 1);
      THTensor_(addmm)(gradWeight, 1, gradWeight, scale, tgradOutputWindow, inputWindow);
      c10::raw::intrusive_ptr::decref(tgradOutputWindow);
    }
  }
  else
  {
    THTensor *gradOutputSample = THTensor_(new)();
    THTensor *inputSample = THTensor_(new)();
    int nBatchFrame = input->size(0);

    for(i = 0; i < nBatchFrame; i++)
    {
      THTensor_(select)(gradOutputSample, gradOutput, 0, i);
      THTensor_(select)(inputSample, input, 0, i);
      int nOutputSampleFrame = nOutputFrame;

      /* bias first */
      for(k = 0; k < nOutputFrame; k++)
      {
        THTensor_(select)(gradOutputWindow, gradOutputSample, 0, k);
        THTensor_(cadd)(gradBias, gradBias, scale, gradOutputWindow);
      }

      /* ouch */
      for(k = 0; nOutputSampleFrame > 0; k++)
      {
        int64_t outputFrameStride = (kW-1)/dW+1;
        int64_t inputFrameStride = outputFrameStride*dW;
        int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THTensor_(setStorage2d)(inputWindow, THTensor_getStoragePtr(inputSample),
                                inputSample->storage_offset()+k*dW*inputSample->size(1),
                                nFrame, inputFrameStride*inputSample->size(1),
                                kW*inputSample->size(1), 1);

        THTensor_(setStorage2d)(gradOutputWindow, THTensor_getStoragePtr(gradOutputSample),
                                gradOutputSample->storage_offset() + k*gradOutputSample->size(1),
                                nFrame, outputFrameStride*gradOutputSample->size(1),
                                gradOutputSample->size(1), 1);

        THTensor *tgradOutputWindow = THTensor_(new)();
        THTensor_(transpose)(tgradOutputWindow, gradOutputWindow, 0, 1);
        THTensor_(addmm)(gradWeight, 1, gradWeight, scale, tgradOutputWindow, inputWindow);
        c10::raw::intrusive_ptr::decref(tgradOutputWindow);
      }
    }
    c10::raw::intrusive_ptr::decref(gradOutputSample);
    c10::raw::intrusive_ptr::decref(inputSample);
  }

  c10::raw::intrusive_ptr::decref(gradOutputWindow);
  c10::raw::intrusive_ptr::decref(inputWindow);
  c10::raw::intrusive_ptr::decref(gradOutput);
  c10::raw::intrusive_ptr::decref(input);

}

#endif
