#include "torch/csrc/jit/script/init.h"

#include "torch/csrc/Device.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/Layout.h"
#include "torch/csrc/jit/import.h"
#include "torch/csrc/jit/script/compiler.h"

#include "torch/csrc/jit/python_tracer.h"
#include "torch/csrc/jit/pybind_utils.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/passes/to_batch.h"
#include "torch/csrc/jit/function_schema.h"
#include "torch/csrc/jit/script/parser.h"

#include <torch/csrc/api/include/torch/detail/ordered_dict.h>

#include <ATen/ATen.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <pybind11/functional.h>


namespace torch {
namespace jit {
namespace script {

using ResolutionCallback = std::function<py::function(std::string)>;

// The visibility attribute is to avoid a warning about storing a field in the
// struct that has a different visibility (from pybind) than the struct.
#ifdef _WIN32
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

static std::string typeString(py::handle h) {
  return py::str(h.get_type().attr("__name__"));
}

inline std::shared_ptr<SugaredValue> toSimple(Value* v) {
  return std::make_shared<SimpleValue>(v);
}

// NB: This should be the single entry-point for instantiating a SugaredValue
// from a Python object. If you are adding support for converting a new Python
// type, *add it in this function's implementation*.
std::shared_ptr<SugaredValue> toSugaredValue(
    py::object obj,
    Method& m,
    SourceRange loc,
    bool is_constant = false,
    bool is_submodule = false);

struct VISIBILITY_HIDDEN PythonValue : public SugaredValue {
  PythonValue(py::object self)
  : self(std::move(self)) {}

  FunctionSchema getSchema(const size_t n_args, const size_t n_binders) {
    auto annotations = py::module::import("torch.jit.annotations");
    auto signature = annotations.attr("get_signature")(self);
    std::vector<Argument> args, rets;
    // We may mutate this if we can determine the number of args from Python
    // introspection.
    size_t actual_n_args = n_args;
    if (!signature.is_none()) {
      std::vector<TypePtr> arg_types;
      TypePtr ret_type;
      std::tie(arg_types, ret_type) = py::cast<std::pair<std::vector<TypePtr>, TypePtr>>(signature);
      args.reserve(arg_types.size());
      size_t idx = 0; // Fake argument names by putting in the index
      for (auto &arg_type : arg_types) {
        args.push_back(Argument(std::to_string(idx++), std::move(arg_type), {}, {}, false));
      }
      rets.push_back(Argument("0", std::move(ret_type), {}, {}, false));
    } else {
      // Create a default signature using what information we have

      // First see if we can introspect the number of function parameters
      // irrespective of the presence of explicit type annotations
      auto num_params = annotations.attr("get_num_params")(self);
      if (!num_params.is_none()) {
        // Return a signature with the correct number of params according to the
        // Python function. The error handling in call() will catch any mismatch
        // later.
        actual_n_args = py::cast<size_t>(num_params);
      }
      // Construct the default signature: all arguments and returns will be
      // DynamicType
      args.reserve(actual_n_args);
      for (size_t i=0; i < actual_n_args; ++i) {
        args.push_back(Argument(std::to_string(i), DynamicType::get(), {}, {}, false));
      }
      TypePtr ret_type = DynamicType::get();
      if(n_binders != 1) {
        std::vector<TypePtr> tuple_values(n_binders, ret_type);
        ret_type = TupleType::create(std::move(tuple_values));
      }
      rets.push_back(Argument("0", ret_type, {}, {}, false));
    }
    return FunctionSchema("", std::move(args), std::move(rets));
  }

  // call it like a function, e.g. `outputs = this(inputs)`
  virtual std::shared_ptr<SugaredValue> call(SourceRange loc, Method & m, at::ArrayRef<NamedValue> inputs_, at::ArrayRef<NamedValue> attributes, size_t n_binders) override {
    auto inputs = toValues(*m.graph(), inputs_);
    auto schema = getSchema(inputs.size(), n_binders);

    std::stringstream failure_messages;
    at::optional<MatchedSchema> matched_schema =
      tryMatchSchema(schema, loc, *m.graph(), inputs_, attributes, failure_messages, /*conv_tensor_to_num*/true);
    if (!matched_schema)
      throw ErrorReport(loc) << failure_messages.str();

    // Release the function object so we can wrap it in a PythonOp
    py::object func = self;
    std::string cconv(inputs.size(), 'd');
    Node* new_node = m.graph()->insertNode(m.graph()->createPythonOp(
      THPObjectPtr(func.release().ptr()), cconv, {}));
    new_node->setSourceLocation(std::make_shared<SourceRange>(loc));
    for(auto &i : matched_schema->inputs)
      new_node->addInput(i);

    std::vector<Value*> outputs;
    for(auto & ret_arg : matched_schema->return_types) {
      outputs.push_back(new_node->addOutput()->setType(ret_arg));
    }
    return std::make_shared<SimpleValue>(packOutputs(*m.graph(), outputs));
  }

  virtual std::string kind() const override {
    std::stringstream ss;
    ss << "python value of type '" << typeString(self) << "'";
    return ss.str();
  }

protected:

  py::object getattr(SourceRange loc, const std::string& name) {
    try {
      return py::getattr(self, name.c_str());
    } catch (py::error_already_set& e) {
      throw ErrorReport(loc) << "object has no attribute " << name;
    }
  }

  py::object self;
};

struct VISIBILITY_HIDDEN PythonModuleValue : public PythonValue {
  explicit PythonModuleValue(py::object mod) : PythonValue(mod) {}

  std::shared_ptr<SugaredValue> attr(
      SourceRange loc,
      Method& m,
      const std::string& field) override {
    py::object member = getattr(loc, field);
    // note: is_constant = true because we consider that global properties
    // on modules like math.pi or torch.float to be constants
    // eventhough it is possible, though rare, for someone to mutate them
    return toSugaredValue(member, m, loc, /*is_constant=*/true);
  }
};

struct VISIBILITY_HIDDEN ConstantPythonTupleValue : public PythonValue {
  explicit ConstantPythonTupleValue(py::object tup) : PythonValue(tup) {}
  std::vector<std::shared_ptr<SugaredValue>> asTuple(SourceRange loc, Method& m, at::optional<size_t> size_hint={}) override {
    py::tuple tup = self;
    std::vector<std::shared_ptr<SugaredValue>> result;
    result.reserve(tup.size());
    for (size_t i = 0; i < tup.size(); ++i) {
      result.push_back(toSugaredValue(tup[i], m, loc, true));
    }
    return result;
  }
};

// defines how modules/methods behave inside the script subset.
// for now this does not have any interaction with python.
// in the future, we will add the ability to resolve `self.foo` to python
// {functions, modules, contants} so this SugaredValue is defined here
// anticipating we will eventually need to replace Module with a py::object
// holding the actual nn.Module class.

// defines how a method obtained from a module behaves in script
struct MethodValue : public SugaredValue {
  MethodValue(std::shared_ptr<Module> module, Method& method)
  : module(std::move(module)) //insurance that method stays alive
  , method(method) {}
  std::string kind() const override {
    return "method";
  }
  virtual std::shared_ptr<SugaredValue> call(SourceRange loc, Method & caller, at::ArrayRef<NamedValue> inputs, at::ArrayRef<NamedValue> attributes, size_t n_binders) override {
    return std::make_shared<SimpleValue>(packOutputs(*caller.graph(), caller.emit_call_to(loc, method, inputs, attributes)));
  }
private:
  std::shared_ptr<Module> module;
  Method& method;

};


struct ModuleValue : public SugaredValue {
  ModuleValue(std::shared_ptr<Module> module)
  : module(std::move(module)) {}

  virtual std::string kind() const override {
    return "module";
  }

  // select an attribute on it, e.g. `this.field`
  virtual std::shared_ptr<SugaredValue> attr(SourceRange loc, Method & m, const std::string& field) override {
    if(NamedModule* v = module->find_module(field)) {
      return std::make_shared<ModuleValue>(v->module);
    } else if(Method* v = module->find_method(field)) {
      return std::make_shared<MethodValue>(module, *v);
    } else if(NamedParameter* v = module->find_parameter(field)) {
      return std::make_shared<SimpleValue>(m.get_or_add_parameter(v->slot()));
    }
    // This can also be a call to a non-script module, or a plain
    // python method. If so return this as a python value.
    py::object py_module = py::cast(module);
    if(py::object attr = py::getattr(py_module, field.c_str(), py::none())) {
      if (py::isinstance<py::function>(attr) ||
          py::isinstance(attr, py::module::import("torch.nn").attr("Module")) ||
          py_module.attr("_constants_set").contains(field.c_str())) {
        return toSugaredValue(attr, m, loc, true);
      } else {
        throw ErrorReport(loc) << "attribute '" << field << "' of type '" << typeString(attr) << "' is not usable in a script method (did you forget to add it __constants__?)";
      }
    }
    throw ErrorReport(loc) << "module has no attribute '" << field << "'";
  }

  // call module.forward
  virtual std::shared_ptr<SugaredValue> call(SourceRange loc, Method & caller, at::ArrayRef<NamedValue> inputs, at::ArrayRef<NamedValue> attributes, size_t n_binders) override {
    return attr(loc, caller, "forward")->call(loc, caller, inputs, attributes, n_binders);
  }

  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(SourceRange loc, Method& m, at::optional<size_t> size_hint={}) override {
    py::object py_module = py::cast(module);
    if(!py::isinstance(py_module, py::module::import("torch.jit").attr("_ConstModuleList")))
      return SugaredValue::asTuple(loc, m, size_hint);
    std::vector<std::shared_ptr<SugaredValue>> result;
    for(py::handle module : py_module) {
      py::object obj = py::reinterpret_borrow<py::object>(module);
      result.push_back(toSugaredValue(
          obj,
          m,
          loc,
          /*is_constant =*/false,
          /*is_submodule =*/true));
    }
    return result;
  }

 private:
  std::shared_ptr<Module> module;
};

std::shared_ptr<SugaredValue> toSugaredValue(
    py::object obj,
    Method& m,
    SourceRange loc,
    bool is_constant,
    bool is_submodule) {
  // directly create SimpleValues when possible, because they are first-class
  // and can be re-assigned. Otherwise, this would be invalid:
  // f = python_constant
  // while ...
  //   f = f + 1
  auto& g = *m.graph();
  if (is_constant) {
    if (py::isinstance<py::int_>(obj)) {
      return toSimple(g.insertConstant(py::cast<int64_t>(obj), loc));
    } else if (py::isinstance<py::float_>(obj)) {
      return toSimple(g.insertConstant(py::cast<float>(obj), loc));
    } else if (py::isinstance<py::bool_>(obj)) {
      return toSimple(g.insertConstant(py::cast<bool>(obj), loc));
    } else if (THPDevice_Check(obj.ptr())) {
      auto device = (THPDevice*)obj.ptr();
      std::vector<int64_t> v = {static_cast<int64_t>(device->device.type()),
                                device->device.index()};
      return toSimple(g.insertConstant(std::move(v)));
    } else if (THPLayout_Check(obj.ptr())) {
      auto layout = (THPLayout*)obj.ptr();
      const auto v = static_cast<int64_t>(layout->layout);
      return toSimple(g.insertConstant(v, loc));
    } else if (THPDtype_Check(obj.ptr())) {
      auto dtype = (THPDtype*)(obj.ptr());
      const auto v = static_cast<int64_t>(dtype->scalar_type);
      return toSimple(g.insertConstant(v, loc));
    } else if (py::isinstance<py::tuple>(obj)) {
     return std::make_shared<ConstantPythonTupleValue>(obj);
    }
  }
  if (py::isinstance<Module>(obj)) {
    auto mod = py::cast<std::shared_ptr<Module>>(obj);
    // In the case that this Python object is not a submodule, inline *ONLY
    // PURE* ScriptModules. This allows us to call arbitrary @script functions
    // within a scripting context while still enforcing that parameters from
    // stateful submodules are properly accounted for.
    if (!is_submodule && mod->get_parameters().size() != 0) {
      throw ErrorReport()
          << "Attempted to inline a Module with parameters. "
             "Stateful modules to be inlined must be submodules of the callee.";
    }
    return std::make_shared<ModuleValue>(mod);
  } else if (py::isinstance<py::module>(obj)) {
    return std::make_shared<PythonModuleValue>(obj);
  }
  py::object builtin_name = py::module::import("torch.jit").attr("_find_builtin")(obj);
  if (!builtin_name.is_none()) {
    return std::make_shared<BuiltinFunction>(
        Symbol::fromQualString(py::str(builtin_name)), at::nullopt);
  }
  return std::make_shared<PythonValue>(obj);
}

py::object unpackVariableTensorList(std::vector<at::Tensor> outputs) {
  // if we don't tell pybind these are variables it chokes on the
  // conversion.
  // TODO: fix conversions to be sane and make sure this works.
  if (outputs.size() == 0) {
    return py::none();
  } else if (outputs.size() == 1) {
    return py::cast(autograd::as_variable_ref(outputs[0]));
  } else {
    py::tuple tuple(outputs.size());
    for(size_t i = 0; i < outputs.size(); i++) {
      tuple[i] = py::cast(autograd::as_variable_ref(outputs[i]));
    }
    return tuple;
  }
}

static void gatherParametersAndBuffers(std::vector<at::Tensor*> & values, const Module & m) {
  for(auto & param : m.get_parameters()) {
    values.push_back(param->slot());
  }
  for(const auto & sub : m.get_modules()) {
    gatherParametersAndBuffers(values, *sub->module);
  }
}

Resolver pythonResolver(ResolutionCallback rcb) {
  return [=](const std::string& name,
             Method& m,
             const SourceRange& loc) -> std::shared_ptr<SugaredValue> {
    AutoGIL ag;
    py::object obj = rcb(name);
    if (obj.is(py::none())) {
      return nullptr;
    }
    return toSugaredValue(obj, m, loc);
  };
}

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // torch.jit.ScriptModule is a subclass of this C++ object.
  // Methods here are prefixed with _ since they should not be
  // public.
  py::class_<Module, std::shared_ptr<Module>>(m, "ScriptModule")
      .def(py::init<>())
      .def("save", &Module::save)
      .def("_set_optimized", &Module::set_optimized)
      .def(
          "_define",
          [](std::shared_ptr<Module> m,
             const std::string& script,
             ResolutionCallback rcb, bool has_self) {
            auto self = has_self ? std::make_shared<ModuleValue>(m) : nullptr;
            return defineMethodsInModule(*m, script, pythonResolver(rcb), self);
          })
      .def("_create_methods", [](std::shared_ptr<Module> m, const std::vector<Def>& defs, const std::vector<ResolutionCallback>& rcbs) {
        std::vector<Resolver> resolvers;
        for(auto & callback : rcbs) {
          resolvers.push_back(pythonResolver(callback));
        }
        defineMethodsInModule(
          *m,
          defs,
          resolvers,
          std::make_shared<ModuleValue>(m));
      })
      .def("_get_method",
      [](Module& self, const std::string& name) -> const Method& {
        return self.get_method(name);
      }, py::return_value_policy::reference_internal)
      .def("_register_parameter", &Module::register_parameter)
      .def("_register_module", &Module::register_module)
      .def("_set_parameter", &Module::set_parameter)
      .def("_get_parameter", &Module::get_parameter)
      .def("_get_module", &Module::get_module)
      .def("_get_modules", [](Module& self) -> py::tuple {
        auto & modules = self.get_modules();
        py::tuple result(modules.size());
        for(size_t i = 0; i < modules.size(); ++i) {
          auto & item = modules[i];
          result[i] = std::make_pair(item.key, item.value);
        }
        return result;
      })
      .def("_get_parameters", [](Module& self) -> py::tuple {
        auto & parameters = self.get_parameters();
        py::tuple result(parameters.size());
        for(size_t i = 0; i < parameters.size(); ++i) {
          auto & p = parameters[i];
          py::tuple r(3);
          result[i] = std::make_tuple(
            p.key,
            autograd::as_variable_ref(*p->slot()),
            p->is_buffer);

        }
        return result;
      })
      .def("_has_parameter", [](Module& self, const std::string& name) {
        if(auto r = self.find_parameter(name)) {
          return !r->is_buffer;
        }
        return false;
      })
      .def("_has_buffer", [](Module& self, const std::string& name) {
        if(auto r = self.find_parameter(name)) {
          return r->is_buffer;
        }
        return false;
      })
      .def("_has_module", [](Module& self, const std::string& name) {
        return bool(self.find_module(name));
      })
      .def("_has_method", [](Module& self, const std::string& name) {
        return bool(self.find_method(name));
      })
      .def("_method_names", [](Module& self) {
        using Item = torch::detail::OrderedDict<std::string, std::unique_ptr<Method>>::Item;
        return fmap(self.get_methods(), [](const Item & item) {
          return (*item)->name();
        });
      })
      .def("_create_method_from_graph", [](
        Module& self,
        const std::string& name,
        std::shared_ptr<Graph> graph
      ){
        std::vector<at::Tensor*> parameters;
        self.create_method(name, std::move(graph), std::move(parameters));
      })
      .def("_create_method_from_trace", [](
        Module& self,
        const std::string& name,
        py::function func,
        py::tuple input_tuple) {
          // prereq: Module's buffers and parameters are unique
          // this was ensured in python before calling this function
          std::vector<at::Tensor*> parameters;
          gatherParametersAndBuffers(parameters, self);
          Stack inputs = toStack(input_tuple);
          for(at::Tensor* param : parameters) {
            inputs.emplace_back(*param);
          }
          auto graph = tracer::createGraphByTracing(func, inputs, input_tuple.size());
          self.create_method(name, std::move(graph), std::move(parameters));
      })
      .def("graph_for", [](Module& self, py::args args, py::kwargs kwargs) {
        if (self.find_method("forward")) {
          Method & m = self.get_method("forward");
          return m.graph_for(
              createStackForSchema(m.getSchema(), std::move(args), std::move(kwargs)));
        }
        throw std::runtime_error("Attempted to call graph_for on a Module without a compiled forward()");
      })
      .def("get_debug_state", [](Module& self) {
        if (self.find_method("forward")) {
          Method & m = self.get_method("forward");
          return m.getDebugState();
        }
        throw std::runtime_error("Attempted to call get_debug_state on a Module without a compiled forward()");
      })
      .def("debug_disable_autodiff_subgraph_inlining", [](Module& self) {
        if (self.find_method("forward")) {
          Method & m = self.get_method("forward");
          m.debugDisableAutodiffSubgraphInlining();
        }
      })
      .def("forward", [](Module& self, py::args args, py::kwargs kwargs) {
        // We implement this in C++ to avoid incurring the pybind11 dispatch
        // overhead twice: once to call into the method lookup for "forward"
        // and once to actually invoke the method.
        //
        // There is a thin wrapper on top of this method in the C++ version of
        // ScriptModule.
        return invokeScriptMethodFromPython(self.get_method("forward"), std::move(args), std::move(kwargs));
      });

  py::class_<Method>(m, "ScriptMethod", py::dynamic_attr())
    .def("graph", [&](Method& self) {
      return self.graph();
    })
    .def("__call__", invokeScriptMethodFromPython)
    .def_property_readonly("graph", [](Method& m) {
      return m.graph();
    })
    .def("propagate_shapes", &Method::propagate_shapes)
    .def("propagate_and_assign_input_and_output_shapes", &Method::propagate_and_assign_input_and_output_shapes)
    .def("params", &Method::params)
    .def("graph_for", [](Method& self, py::args args, py::kwargs kwargs) {
      return self.graph_for(createStackForSchema(self.getSchema(), std::move(args), std::move(kwargs)));
    })
    .def("forward_schema", [](Method &self, Def &def, bool is_method) {
      auto schema = extractSchemaFromDef(def, is_method);
      self.setSchema(schema);
    })
    .def("debug_disable_autodiff_subgraph_inlining", &Method::debugDisableAutodiffSubgraphInlining)
    .def("pretty_print_schema", &Method::pretty_print_schema);

  m.def("_jit_script_compile", [](const Def &def, ResolutionCallback rcb) {
    return compileFunction(def, pythonResolver(rcb));
  });

  m.def("parse_type_comment", [](const std::string& comment) {
    Parser p(comment);
    return Decl(p.parseTypeComment(true));
  });

  m.def("merge_type_from_type_comment", &mergeTypesFromTypeComment);
  m.def("import_ir_module", import_ir_module);
}

} // namespace script
} // namespace jit
} // namespace torch
