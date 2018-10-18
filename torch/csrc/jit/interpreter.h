#pragma once
#include <memory>
#include <vector>
#include "c10/util/Optional.h"

#include "torch/csrc/WindowsTorchApiMacro.h"

namespace at {
  class Tensor;
}
namespace c10 {
struct IValue;
}
namespace torch { namespace jit {

// The interpreter run Graphs with Tensor inputs and Tensor outputs
// a separate component in the autograd handles unwrapping and wrapping
// variable objects for use in the interpreter.

struct Node;
struct GraphExecutor;
struct CodeImpl;
struct InterpreterStateImpl;
struct Graph;
struct Node;
using Stack = std::vector<c10::IValue>;

struct TORCH_API Code {
  Code()
    : pImpl(nullptr) {}
  Code(std::shared_ptr<Graph>& graph);
  ~Code();

  const std::vector<GraphExecutor*>& grad_executors();

  explicit operator bool() const {
    return pImpl != nullptr;
  }

private:
  std::shared_ptr<CodeImpl> pImpl;
  friend struct InterpreterStateImpl;
  friend std::ostream & operator<<(std::ostream & out, const Code & code);
};

struct InterpreterState {
  InterpreterState(const Code & code);
  void run(Stack & stack);
  ~InterpreterState();
  // create a copy of InterpreterState with its current state
  // used when retain_graph=True
  InterpreterState clone() const;
private:
  InterpreterState(InterpreterStateImpl * pImpl);
  std::shared_ptr<InterpreterStateImpl> pImpl;
};

}}
