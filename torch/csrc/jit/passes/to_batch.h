#pragma once

#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

class ToBatch {
private:
  // mapping from tensor in original graph to {data, mask, dims} in new graph
  std::unordered_map<Value*, std::vector<Value*>> batch_map;
public:
  static std::unordered_map<std::string, std::shared_ptr<Graph>> batch_operator_table;
  TORCH_API void toBatch(Block* block, Block* res_block);
};

TORCH_API std::shared_ptr<Graph> to_batch_graph(std::shared_ptr<Graph>& graph);
TORCH_API void initRegisterBatchOpsBindings(PyObject* module);
}}
