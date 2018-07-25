#include "torch/csrc/jit/graph_executor.h"

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/passes/batch_mm.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/erase_number_types.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/inplace_check.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/remove_expands.h"
#include "torch/csrc/jit/passes/decompose_addmm.h"
#include "torch/csrc/jit/passes/specialize_undef.h"
#include "torch/csrc/jit/passes/loop_unrolling.h"
#include "torch/csrc/jit/passes/lower_grad_of.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/ivalue.h"

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/jit/script/compiler.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch { namespace jit {

namespace {

using tensor_list = std::vector<at::Tensor>;
using Variable = autograd::Variable;
using autograd::variable_list;

// this type is in ExecutionPlan to run its Gradient if it is
// specified. It has a list of inputs captured by ExecutionPlan that
// it concats with inputs to form the full set of inputs to graph.
// see struct Gradient for a description of how the derivative graph
// is constructed and what variables are captured.
struct ExecutionPlanAutogradFunction : public autograd::Function {
  ExecutionPlanAutogradFunction(GraphExecutor graph, size_t capture_size)
  : graph(std::move(graph)) {
    captures.reserve(capture_size);
  }
  virtual variable_list apply(variable_list&& inputs) override {
    // TODO: expensive copies here to convert to/from tensor_list
    // TODO: because inputs is passed by const reference there is no
    // way to release tensors incrementally as this runs
    variable_tensor_list all_inputs;
    all_inputs.reserve(captures.size() + inputs.size());
    all_inputs.insert(all_inputs.end(), inputs.begin(), inputs.end());
    for(auto & sv : captures) {
      all_inputs.push_back(sv.unpack(this->shared_from_this()));
    }
    auto tensors = graph.run(std::move(all_inputs));
    // TODO: another copy that needs to be removed
    return autograd::variable_list(tensors.begin(), tensors.end());
  }
private:
  friend struct ExecutionPlan;
  GraphExecutor graph;
  std::vector<autograd::SavedVariable> captures;
};


// helper to run interpreter on variables until we switch
// everything to IValue
inline variable_tensor_list runOneStage(const Code & code, variable_tensor_list inputs) {
  std::vector<IValue> stack(inputs.begin(), inputs.end());
  InterpreterState(code).runOneStage(stack);
  return variable_tensor_list(fmap(stack, [](IValue& v) {
    return std::move(v).toTensor();
  }));
}

// an optimized way of executing the subgraph computed directly on
// tensors rather than Variables.
// This will unwrap Variables, run the plan, and re-wrap them.
// It can optionally also have a gradient which is hooked up
// to the output Variables if present.
struct ExecutionPlan {
  ExecutionPlan(std::shared_ptr<Graph>& graph)
      : f(graph), graph(graph) {}
  ExecutionPlan(std::shared_ptr<Graph>& graph, Gradient grad)
      : f(graph),
        graph(graph),
        grad(std::move(grad)),
        grad_executor(this->grad.df) {}

  variable_tensor_list run(variable_tensor_list&& stack) const {
    if(grad) {
      return runWithGrad(std::move(stack));
    }
    return runOneStage(f, std::move(stack));
  }
  std::shared_ptr<Graph> get_graph() const {
    return graph;
  }

  ExecutionPlanState getDebugState() {
    ExecutionPlanState state;
    state.f = &f;
    state.graph = graph.get();
    if (grad) {
      state.grad = &grad;
      state.grad_executor = std::unique_ptr<GraphExecutorState>(
          new GraphExecutorState(grad_executor.getDebugState()));
    } else {
      state.grad = nullptr;
      state.grad_executor.reset();
    }
    return state;
  }

private:
  // note: should be inplace to avoid allocations, but we have to switch from
  // a list of tensor to a list of ivalues
  std::vector<IValue> unwrapVariables(variable_tensor_list && list) const {
    return fmap(list, [](const Variable& v) -> IValue {
      return v.defined() ? autograd::as_variable_ref(v).detach() : at::Tensor();
    });
  }
  // note: should be inplace to avoid allocations, but we have to switch from
  // a list of tensor to a list of ivalues
  variable_tensor_list wrapTensors(tensor_list && list) const {
    for(auto & v : list) {
      v = autograd::make_variable(v, /*requires_grad=*/false);
    }
    return variable_tensor_list(std::move(list));
  }
  // Capture (save) inputs that would be required to subsequently run backwards
  void captureInputs(ExecutionPlanAutogradFunction & grad_fn, variable_tensor_list & inputs) const {
    for(auto offset : grad.df_input_captured_inputs) {
      grad_fn.captures.emplace_back(autograd::as_variable_ref(inputs[offset]), false);
    }
  }
  void captureOutputs(ExecutionPlanAutogradFunction & grad_fn, variable_tensor_list & outputs) const {
    for(auto offset : grad.df_input_captured_outputs) {
      grad_fn.captures.emplace_back(autograd::as_variable_ref(outputs[offset]), true);
    }
  }

  variable_tensor_list runWithGrad(variable_tensor_list&& inputs) const {
    auto grad_fn = std::make_shared<ExecutionPlanAutogradFunction>(grad_executor,
      grad.df_input_captured_inputs.size() + grad.df_input_captured_outputs.size());
    // hook up the outputs of df to the gradient functions of the inputs that require
    // gradients
    for(auto idx : grad.df_output_vjps) {
      auto & v = autograd::as_variable_ref(inputs[idx]);
      grad_fn->add_next_edge(v.gradient_edge());
    }
    captureInputs(*grad_fn, inputs);

    auto stack = unwrapVariables(std::move(inputs));
    InterpreterState(f).runOneStage(stack);
    variable_tensor_list outputs(
        fmap(stack, [](IValue& v) { return std::move(v).toTensor(); }));

    // hookup the gradients for the output tensors that require gradients
    // to the inputs to our gradient function df
    // TODO - XXX - if any output is the same tensor multiple times, views have to be
    // setup here. We need to refactor autograd until it is safe for
    // tensors to be constructed without all the viewing infrastructure.
    // this is currently intentionally not done here so we can get an idea of our
    // perf before introducing overhead for correctness
    for(auto idx : grad.df_input_vjps) {
      // Note: we have to set this up in place, or we have to throw away and
      // reallocate variables that were already created in wrapTensors. We
      // should add an API for this.
      auto& output = autograd::as_variable_ref(outputs[idx]);
      autograd::create_gradient_edge(output, grad_fn);
      output.set_requires_grad(true);
    }
    captureOutputs(*grad_fn, outputs);
    // drop the temporary outputs so that we return the same number of
    // outputs as if we were not also calculating gradient
    outputs.erase(outputs.begin() + grad.f_real_outputs, outputs.end());
    return outputs;
  }
  Code f;
  // optimized graph for debugging and testing
  std::shared_ptr<Graph> graph;
  // description of gradient as a graph
  Gradient grad; // if(grad) is false when this is unused
  // executor for df, including code caches
  GraphExecutor grad_executor;
};

} // anonymous namespace

// a Graph can be created via tracing, or via a language-based frontend
// GraphExecutor runs it. It can run the same graph on many different sizes
// and different requires_grad states, and handles specializations for each situation.
// GraphExecutor is completely unaware of tracing or module parameters to keep the
// tracing concerns separated.
struct GraphExecutorImpl {

  GraphExecutorImpl(std::shared_ptr<Graph> graph, bool optimize, bool symbolically_differentiable)
  : graph(std::move(graph))
  , optimize(optimize)
  , num_inputs(this->graph->inputs().size())
  , symbolically_differentiable(symbolically_differentiable)
  , may_introduce_gradient(calcMayIntroduceGradient(this->graph->block())) {}
  GraphExecutorImpl(std::shared_ptr<Graph> graph, bool optimize)
  : GraphExecutorImpl(graph, optimize, isDifferentiable(*graph)) {}

  // entry point where execution begins
  variable_tensor_list run(variable_tensor_list inputs) {
    if(inputs.size() != num_inputs) {
      std::stringstream ss;
      ss << "expected " << num_inputs << " inputs but got " << inputs.size() << " inputs";
      throw std::runtime_error(ss.str());
    }

    // the tracer has called a graph executor
    // there is no need to optimize, but we do need to splice the graph of
    // this excutor into the trace. Otherwise we might unroll control-flow
    // operations.
    if(tracer::isTracing()) {
      return runTraced(std::move(inputs));
    }

    // this is the fallback pathway, when we cannot differentiate
    if(!optimize || (!symbolically_differentiable && needsGradient(inputs))) {
      return runFallback(std::move(inputs));
    }

    // either we can symbolically differentiate, or we do not need a gradient.
    // go down the route where we treat the inputs as tensors
    // and fully optimize
    auto & implementation = getOrCompile(inputs);
    return implementation.run(std::move(inputs));
  }

  std::shared_ptr<Graph> graphFor(const variable_tensor_list& inputs) const {
    ArgumentSpec spec(autograd::GradMode::is_enabled(), inputs);

    if (!optimize || (!symbolically_differentiable && needsGradient(inputs))) {
      JIT_ASSERTM(autograd_fallback_graph, "No graph found for given inputs");
      return autograd_fallback_graph;
    }

    auto it = plan_cache.find(spec);
    JIT_ASSERTM(it != plan_cache.end(), "No graph found for given inputs");
    return it->second.get_graph();
  }

  GraphExecutorState getDebugState() {
    GraphExecutorState state;
    state.graph = graph.get();
    if (autograd_fallback) {
      state.autograd_fallback = &autograd_fallback;
      state.autograd_fallback_graph = autograd_fallback_graph.get();
    } else {
      state.autograd_fallback = nullptr;
      state.autograd_fallback_graph = nullptr;
    }
    for (auto & entry : plan_cache) {
      state.execution_plans.emplace(entry.first, entry.second.getDebugState());
    }
    return state;
  }

private:
  friend struct GraphExecutor;

  variable_tensor_list runTraced(variable_tensor_list inputs) {
    auto state = tracer::getTracingState();
    auto input_values = fmap(inputs, tracer::getValueTrace);

    ArgumentSpec spec(autograd::GradMode::is_enabled(), inputs);
    auto outputs = runFallback(std::move(inputs));

    auto all_dynamic = [](const at::ArrayRef<Value*> xs) {
      for(Value* x : xs) {
        if(x->type()->kind() != TypeKind::DynamicType)
          return false;
      }
      return true;
    };
    // Traces always have types propagated through them, so we make sure to
    // also propagate types through the graph we are inserting here.
    // However, this->graph itself may already have been generated with
    // tracing and so we only do the type propgation if no concrete types have
    // been set.
    auto local_graph = this->graph;
    if(all_dynamic(local_graph->inputs()) && all_dynamic(local_graph->outputs())) {
      local_graph = this->graph->copy();
      PropagateInputShapes(*local_graph, spec);
    }
    auto output_values = script::inlineCallTo(*state->graph, *local_graph, input_values);

    for(size_t i = 0; i < outputs.size(); ++i) {
      tracer::setValueTrace(outputs[i], output_values[i]);
    }
    return outputs;
  }

  variable_tensor_list runFallback(variable_tensor_list inputs) {
    auto & fb = getOrCreateAutogradFallback();
    return runOneStage(fb, std::move(inputs));
  }

  static bool calcMayIntroduceGradient(Block* b) {
    for(Node* n : b->nodes()) {
      if(n->kind() == prim::PythonOp)
        return true;
      for(Block* bb : n->blocks()) {
        if(calcMayIntroduceGradient(bb))
          return true;
      }
    }
    return false;
  }
  bool needsGradient(const variable_tensor_list & inputs) const {
    if (!autograd::GradMode::is_enabled()) {
      return false;
    }
    if(may_introduce_gradient)
      return true;
    for (const auto & tensor : inputs) {
      if(tensor.defined() && static_cast<const Variable&>(tensor).requires_grad())
        return true;
    }
    return false;
  }

  const Code & getOrCreateAutogradFallback() {
    std::lock_guard<std::mutex> lock(compile_mutex);
    if(autograd_fallback) {
      return autograd_fallback;
    }
    auto graph_ = graph->copy();
    runRequiredPasses(graph_);
    if(optimize) {
      if(!symbolically_differentiable)
        CreateAutodiffSubgraphs(*graph_);
      runOptimization(graph_, /*graphMustSupportVariables=*/true);
    }
    autograd_fallback_graph = graph_;
    autograd_fallback = Code(graph_);
    return autograd_fallback;
  }
  const ExecutionPlan & getOrCompile(const variable_tensor_list & inputs) {
    // outside lock guard, to minimize the time holding the lock on the fast path
    // ArgumentSpec even computes its hashCode here.
    ArgumentSpec spec(autograd::GradMode::is_enabled(), inputs);
    {
      std::lock_guard<std::mutex> lock(compile_mutex);
      auto it = plan_cache.find(spec);
      if(it != plan_cache.end())
        return it->second;
      auto plan = compileSpec(spec);
      auto r = plan_cache.emplace(std::move(spec), std::move(plan));
      return r.first->second;
    }
  }

  bool argumentSpecRequiresGradient(const ArgumentSpec & spec) {
    for(size_t i = 0; i < spec.size(); ++i) {
      if(spec.tensorInfo(i).requires_grad())
        return true;
    }
    return false;
  }

  ExecutionPlan compileSpec(const ArgumentSpec & spec) {
    auto graph_ = graph->copy();

    specializeToSpec(graph_, spec);

    if(!argumentSpecRequiresGradient(spec)) {
      runOptimization(graph_, /*graphMustSupportVariables=*/false);
      return ExecutionPlan(graph_);
    }
    JIT_ASSERT(symbolically_differentiable);

    std::vector<bool> requires_grads;
    requires_grads.reserve(spec.size());
    for(size_t i = 0; i < spec.size(); i++)
      requires_grads.push_back(spec.tensorInfo(i).requires_grad());

    Gradient gradient = differentiate(graph_, requires_grads);
    graph_ = gradient.f;
    runOptimization(graph_, /*graphMustSupportVariables=*/false);
    return ExecutionPlan(graph_, std::move(gradient));
  }
  // the unoptimized starting graph
  // this is never mutated
  std::shared_ptr<Graph> graph;

  // true - do everything we can to make this graph run fast
  // false - do not modifiy the graph at all and just use the interpreter
  // to run the graph. Useful for debugging correctness issues in the implementation
  bool optimize;
  size_t num_inputs;

  // GraphExecutor optimizes more aggresively when we _know_ the graph will be
  // symbolically differentiable.
  bool symbolically_differentiable;

  // some ops, including python operations, can intorduce requires_grad=True
  // variables even though no inputs to this graph are availiable, if
  // the graph includes those operators then needGradient must be true
  // regardles of input state.
  bool may_introduce_gradient;

  // when this graph has some parts that are not symbolically_differentable,
  // but some input does require a derivative, we create and use autograd_fallback,
  // which wraps up the fully differentiable subgraphs, and then runs the outer
  // graph through autograd.
  // Since we can't optimize black box functions anyway, there is only one fallback path,
  // and it must work on all sizes (so no optimizations that inspect sizes can run on it)
  std::shared_ptr<Graph> autograd_fallback_graph;
  Code autograd_fallback;

  // optimizable code paths, used when we can differentiate or when no derivative is needed
  // Spec describes input conditions, Plan describes how to execute them.
  std::unordered_map<ArgumentSpec, ExecutionPlan> plan_cache;

  // GraphExecutor can be accessed from  multiple thread so
  // anytime we are checking or updating the autograd_fallback or
  // plan_cache, we must hold the compile mutex.
  // along the fast path (no compilation) code should
  // hold this for as little time as possible.
  std::mutex compile_mutex;
};

GraphExecutor::GraphExecutor(std::shared_ptr<Graph> graph, bool optimize)
: pImpl(new GraphExecutorImpl(std::move(graph), optimize)) {}

GraphExecutor::GraphExecutor(std::shared_ptr<Graph> graph, bool optimize, bool symbolically_differentiable)
: pImpl(new GraphExecutorImpl(std::move(graph), optimize, symbolically_differentiable)) {}

variable_tensor_list GraphExecutor::run(variable_tensor_list && inputs) {
  return pImpl->run(std::move(inputs));
}

std::shared_ptr<Graph> GraphExecutor::graph() const {
  return pImpl->graph;
}

std::shared_ptr<Graph> GraphExecutor::graphFor(const variable_tensor_list& inputs) const {
  return pImpl->graphFor(inputs);
}

GraphExecutorState GraphExecutor::getDebugState() {
  return pImpl->getDebugState();
}


void runRequiredPasses(const std::shared_ptr<Graph>& g)  {
  LowerGradOf(*g);
  // implicit inserted expand nodes are not necessarily always valid
  // when used inside script methods that might have unstable shapes
  // we remove the implicitly created ones, and have shape analysis
  // add valid expand nodes when the shapes are stable
  RemoveExpands(g);
}

void specializeToSpec(const std::shared_ptr<Graph>& graph_, const ArgumentSpec& spec) {
  // clean up GradOf and AutogradAdd nodes
  // this must be first because later passes do not know what GradOfs are
  std::vector<bool> defined;
  for(size_t i = 0; i < spec.size(); ++i) {
    defined.push_back(spec.tensorInfo(i).defined());
  }
  specializeUndef(*graph_, defined);

  // required passes shared with autograd fallback
  runRequiredPasses(graph_);

  // Decompose addmm nodes to add + mm, so expands can be inserted and
  // gradients accumulated on the backward pass
  //
  // In the future, if we need more passes like this, we should convert this
  // into a generic canonicalization pass.
  DecomposeAddmm(graph_);
  // clean up dead constants from specialization
  EliminateDeadCode(graph_);
  // calculate all input shapes
  PropagateInputShapes(*graph_, spec);
}

void runOptimization(std::shared_ptr<Graph> & graph, bool graphMustSupportVariables) {

  // these optimizations must run in the presence of variables
  // and when shape information is not statically known.
  EliminateDeadCode(graph);
  CheckInplace(graph);
  EliminateCommonSubexpression(graph);

  if (!graphMustSupportVariables) {
    // These optimizations can introduce operators like FusionGroup that
    // do not work on variables

    // They also may assume that concrete sizes/strides are availiable
    UnrollLoops(graph);

    //TODO: create peephole optimizations that are safe to run
    // when we are using variables, and when we do not know sizes.
    PeepholeOptimize(graph);
    // TODO: remove mandatory size checking in BatchMM, otherwise
    // it works fine on variables.
    BatchMM(graph);
    FuseGraph(graph);
  }
}

}}
