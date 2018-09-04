#include <fstream>
#include <iterator>
#include <string>

#include "binaries/benchmark_helper.h"

using std::make_shared;
using std::map;
using std::string;
using std::vector;

CAFFE2_DEFINE_string(
    backend,
    "builtin",
    "The backend to use when running the model. The allowed "
    "backend choices are: builtin, default, nnpack, eigen, mkl, cuda");

CAFFE2_DEFINE_string(
    init_net,
    "",
    "The given net to initialize any parameters.");
CAFFE2_DEFINE_string(
    input,
    "",
    "Input that is needed for running the network. If "
    "multiple input needed, use comma separated string.");
CAFFE2_DEFINE_string(
    input_dims,
    "",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");
CAFFE2_DEFINE_string(
    input_file,
    "",
    "Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.");
CAFFE2_DEFINE_string(
    input_type,
    "float",
    "Input type when specifying the input dimension."
    "The supported types are float, uint8_t.");
CAFFE2_DEFINE_int(iter, 10, "The number of iterations to run.");
CAFFE2_DEFINE_string(net, "", "The given net to benchmark.");
CAFFE2_DEFINE_string(
    output,
    "",
    "Output that should be dumped after the execution "
    "finishes. If multiple outputs are needed, use comma "
    "separated string. If you want to dump everything, pass "
    "'*' as the output value.");
CAFFE2_DEFINE_string(
    output_folder,
    "",
    "The folder that the output should be written to. This "
    "folder must already exist in the file system.");
CAFFE2_DEFINE_bool(
    run_individual,
    false,
    "Whether to benchmark individual operators.");
CAFFE2_DEFINE_int(
    sleep_before_run,
    0,
    "The seconds to sleep before starting the benchmarking.");
CAFFE2_DEFINE_bool(
    text_output,
    false,
    "Whether to write out output in text format for regression purpose.");
CAFFE2_DEFINE_int(warmup, 0, "The number of iterations to warm up.");
CAFFE2_DEFINE_bool(
    wipe_cache,
    false,
    "Whether to evict the cache before running network.");

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  benchmark(
      argc,
      argv,
      caffe2::FLAGS_backend,
      caffe2::FLAGS_init_net,
      caffe2::FLAGS_input,
      caffe2::FLAGS_input_dims,
      caffe2::FLAGS_input_file,
      caffe2::FLAGS_input_type,
      caffe2::FLAGS_iter,
      caffe2::FLAGS_net,
      caffe2::FLAGS_output,
      caffe2::FLAGS_output_folder,
      caffe2::FLAGS_run_individual,
      caffe2::FLAGS_sleep_before_run,
      caffe2::FLAGS_text_output,
      caffe2::FLAGS_warmup,
      caffe2::FLAGS_wipe_cache);
}
