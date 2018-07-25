#include "torch/csrc/jit/script/init.h"

#include "torch/csrc/Device.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/Layout.h"
#include "torch/csrc/jit/script/compiler.h"

#include "torch/csrc/jit/python_tracer.h"
#include "torch/csrc/jit/pybind_utils.h"
#include "torch/csrc/jit/constants.h"

#include <torch/csrc/api/include/torch/detail/ordered_dict.h>

#include <ATen/ATen.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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

struct VISIBILITY_HIDDEN PythonValue : public SugaredValue {
  PythonValue(py::object self)
  : self(std::move(self)) {}

  std::pair<std::vector<TypePtr>, TypePtr> getFunctionType(size_t n_args, size_t n_binders) {
    auto annotations = py::module::import("torch.jit.annotations");
    return py::cast<std::pair<std::vector<TypePtr>, TypePtr>>(annotations.attr("get_signature")(self, n_args, n_binders));
  }

  // call it like a function, e.g. `outputs = this(inputs)`
  virtual std::shared_ptr<SugaredValue> call(SourceRange loc, Method & m, at::ArrayRef<NamedValue> inputs_, at::ArrayRef<NamedValue> attributes, size_t n_binders) override {
    auto inputs = toValues(inputs_);
    std::vector<TypePtr> arg_types;
    TypePtr ret_type;
    std::tie(arg_types, ret_type) = getFunctionType(inputs.size(), n_binders);

    if (arg_types.size() != inputs.size())
      throw ErrorReport(loc) << "calling a Python function with an incorrect number "
                             << "of arguments: expected " << arg_types.size() << ", but got "
                             << inputs.size();
    for (size_t i = 0; i < arg_types.size(); ++i) {
      if (!inputs[i]->type()->isSubtypeOf(*arg_types[i]))
        throw ErrorReport(loc) << "type mismatch at argument " << i << ": expected "
                               << arg_types[i]->str() << ", but got " << inputs[i]->type()->str();
    }
    // We have to do this check here, because implementation of this function is tightly
    // coupled with the impl for PythonOp in the interpreter. Right now it assumes that
    // all inputs taken from the stack are Tensors, so that's what we have to do.
    ensureTensors(loc, inputs);

    if (attributes.size() > 0)
      throw ErrorReport(loc) << "keyword arguments in Python calls aren't supported";
    Graph& g = *m.graph();

    // this python object might be a @trace or @script function/module
    // if so, inline the graph rather than calling the python

    if(py::isinstance<Module>(self)) {
      Module& mod = py::cast<Module&>(self);
      if (Method * forward = mod.find_method("forward")) {
        // This code path should only get called for Modules that are really
        // wrappers around pure script/traced functions. Modules with parameters
        // should be submodules of the caller, and thus will be represented as
        // ModuleValue and not go through here.
        if (mod.get_parameters().size() != 0) {
          throw ErrorReport(loc) << "Attempted to inline a Module with parameters. "
            "Stateful modules to be inlined must be submodules of the callee.";
        }
        std::vector<torch::jit::NamedValue> named_inputs;
        for (auto inp : inputs)
          named_inputs.push_back(NamedValue(loc, "", inp));
        return packOutputs(*m.graph(), m.emit_call_to(loc, *forward, named_inputs, {}));
      }
    }

    // Release the function object so we can wrap it in a PythonOp
    py::object func = self;
    std::string cconv(inputs.size(), 't');
    Node* new_node = g.insertNode(g.createPythonOp(
      THPObjectPtr(func.release().ptr()), cconv, {}));
    new_node->setSourceLocation(std::make_shared<SourceRange>(loc));
    for(auto i : inputs)
      new_node->addInput(i);

    // This is really dumb, but relaxing the constraints on return types would
    // require us to change the implementation of PythonOps in the interpreter.
    // Note that this effectively makes the return type of Tuple[Tensor] and Tensor
    // equivalent, but the PythonOp impl ends with an optional tuple unpack, so we need
    // to do it.
    std::shared_ptr<TupleType> ret_tuple_type;
    if (ret_type->kind() != TypeKind::TupleType) {
      ret_tuple_type = std::make_shared<TupleType>(std::vector<TypePtr>{ret_type});
    } else {
      ret_tuple_type = std::static_pointer_cast<TupleType>(ret_type);
    }
    for (auto & ret_type_elem : ret_tuple_type->elements()) {
      if (!ret_type_elem->isSubtypeOf(*DynamicType::get())) {
        throw ErrorReport(loc) << "Python functions can currently only return Tensors";
      }
    }

    std::vector<Value*> outputs;
    for(size_t i = 0; i < ret_tuple_type->elements().size(); ++i)
      outputs.push_back(new_node->addOutput());
    return packOutputs(*m.graph(), outputs);
  }

  virtual std::shared_ptr<SugaredValue> attr(SourceRange loc, Method & m, const std::string& field) override;

  virtual std::string kind() const override {
    std::stringstream ss;
    ss << "python value of type '" << typeString(self) << "'";
    return ss.str();
  }

protected:
  bool isBuiltinModule() {
    // XXX: these can't be static, or they will be destructed after the Python interpreter
    // exits and that generally sounds like a bad idea
    py::object torch = py::module::import("torch");
    py::object functional = py::module::import("torch.nn.functional");
    return self.is(torch) || self.is(functional);
  }

  py::object getattr(SourceRange loc, const std::string& name) {
    try {
      return py::getattr(self, name.c_str());
    } catch (py::error_already_set& e) {
      throw ErrorReport(loc) << "object has no attribute " << name;
    }
  }

  py::object self;
};

// by using torch.jit.Const, a user can mark a python value constant
// we then make that value immutable.
// once marked constant, we enable additional behavior such as
// 1. conversion via asValue to a constant Tensor
// 2. unrolling of for loops
struct VISIBILITY_HIDDEN ConstantPythonValue : public PythonValue {
  using PythonValue::PythonValue;
  virtual Value * asValue(SourceRange loc, Method & m) override {

    return PythonValue::asValue(loc, m);
  }
  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(SourceRange loc, Method& m) override {
    if(!py::isinstance<py::tuple>(self))
      return PythonValue::asTuple(loc, m);

    py::tuple tup = self;
    std::vector<std::shared_ptr<SugaredValue>> result;
    for(size_t i = 0; i < tup.size(); ++i) {
      result.push_back(create(loc, m, tup[i]));
    }
    return result;
  }
  static std::shared_ptr<SugaredValue> create(SourceRange loc, Method& m, py::object self) {
    // directly create SimpleValues when possible, because they are first-class
    // and can be re-assigned. Otherwise, this would be invalid:
    // f = python_constant
    // while ...
    //   f = f + 1
    auto& g = *m.graph();
    if(py::isinstance<py::int_>(self)) {
      return toSimple(insertConstant(g, py::cast<int64_t>(self), loc));
    } else if(py::isinstance<py::float_>(self)) {
      return toSimple(insertConstant(g, py::cast<float>(self), loc));
    } else if(py::isinstance<py::bool_>(self)) {
      return toSimple(insertConstant(g, py::cast<bool>(self), loc));
    } else if(THPDevice_Check(self.ptr())) {
      auto device = (THPDevice*) self.ptr();
      std::vector<int64_t> v = {static_cast<int64_t>(device->device.type()), device->device.index()};
      return toSimple(insertConstant(g, std::move(v)));
    } else if(THPLayout_Check(self.ptr())) {
      auto layout = (THPLayout*) self.ptr();
      const auto v = static_cast<int64_t>(layout->layout);
      return toSimple(insertConstant(g, v, loc));
    } else if(THPDtype_Check(self.ptr())) {
      auto dtype = (THPDtype*)(self.ptr());
      const auto v = static_cast<int64_t>(dtype->scalar_type);
      return toSimple(insertConstant(g, v, loc));
    }
    return std::make_shared<ConstantPythonValue>(self);
  }
};

std::shared_ptr<SugaredValue> PythonValue::attr(SourceRange loc, Method & m, const std::string& field) {
  // We generally don't want to allow traversing arbitrary Python objects, but we
  // make an exception for traversing modules because we want to be access
  // torch, torch.nn.functional, and the functions they expose.
  py::object member = getattr(loc, field);
  if (isBuiltinModule()) {
    if(py::isinstance<py::function>(member)) {
      return std::make_shared<BuiltinFunction>(field, at::nullopt);
    }
    //e.g. any tensor attribute objects such as torch.uint8
    if(THPDtype_Check(member.ptr()) ||
       THPLayout_Check(member.ptr()) ||
       THPDevice_Check(member.ptr())) {
      return ConstantPythonValue::create(loc, m, member);
    }
  }
  if (py::isinstance<py::module>(self) && py::isinstance<py::module>(member)) {
    return std::make_shared<PythonValue>(member);
  }
  throw ErrorReport(loc) << "unsupported attribute lookup on " << py::repr(self) << ".";
}

Resolver pythonResolver(ResolutionCallback rcb) {
  return [=](const std::string& name) -> std::shared_ptr<SugaredValue> {
      AutoGIL ag;
      py::object obj = rcb(name);
      if(obj.is(py::none())) {
        return nullptr;
      }
      return std::make_shared<PythonValue>(obj);
  };
}

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
    return packOutputs(*caller.graph(), caller.emit_call_to(loc, method, inputs, attributes));
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
      if(py::isinstance<py::function>(attr) ||
         py::isinstance(attr, py::module::import("torch.nn").attr("Module"))) {
        return std::make_shared<PythonValue>(attr);
      } else if(py_module.attr("_constants_set").contains(field.c_str())) {
        return ConstantPythonValue::create(loc, m, attr);
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

  virtual std::vector<std::shared_ptr<SugaredValue>> asTuple(SourceRange loc, Method& m) override {
    py::object py_module = py::cast(module);
    if(!py::isinstance(py_module, py::module::import("torch.jit").attr("_ConstModuleList")))
      return SugaredValue::asTuple(loc, m);
    std::vector<std::shared_ptr<SugaredValue>> result;
    for(py::handle module : py_module) {
      py::object obj = py::reinterpret_borrow<py::object>(module);
      if(py::isinstance<Module>(obj)) {
        auto r = py::cast<std::shared_ptr<Module>>(obj);
        result.push_back(std::make_shared<ModuleValue>(r));
      } else {
        result.push_back(ConstantPythonValue::create(loc, m, obj));
      }
    }
    return result;
  }

private:
  std::shared_ptr<Module> module;
};


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

py::object runMethodFromPython(Method& m, py::args args) {
  auto inputs = createVariableTensorList(args);
  auto outputs = m.run(std::move(inputs));
  return unpackVariableTensorList(std::move(outputs));
}

void initJitScriptBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  // torch.jit.ScriptModule is a subclass of this C++ object.
  // Methods here are prefixed with _ since they should not be
  // public.
  py::class_<Module, std::shared_ptr<Module>>(m, "ScriptModule")
      .def(py::init<>())
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
        tracer::variable_list inputs) {
          size_t num_inputs = inputs.size();
          // prereq: Module's buffers and parameters are unique
          // this was ensured in python before calling this function
          std::vector<at::Tensor*> parameters;
          gatherParametersAndBuffers(parameters, self);
          for(at::Tensor* param : parameters) {
            inputs.push_back(autograd::as_variable_ref(*param));
          }
          auto graph = tracer::createGraphByTracing(func, std::move(inputs), num_inputs);
          self.create_method(name, std::move(graph), std::move(parameters));
      })
      .def("graph_for", [](Module& self, py::args args) {
        if (self.find_method("forward")) {
          return self.get_method("forward").graph_for(createVariableTensorList(args));
        }
        throw std::runtime_error("Attempted to call graph_for on a Module without a compiled forward()");
      })
      .def("forward", [](Module& self, py::args args) {
        // We implement this in C++ to avoid incurring the pybind11 dispatch
        // overhead twice: once to call into the method lookup for "forward"
        // and once to actually invoke the method.
        //
        // There is a thin wrapper on top of this method in the C++ version of
        // ScriptModule.
        return runMethodFromPython(self.get_method("forward"), args);
      });

  py::class_<Method>(m, "ScriptMethod", py::dynamic_attr())
    .def("graph", [&](Method& self) {
      return self.graph();
    })
    .def("__call__", [](Method& m, py::args args) -> py::object {
      return runMethodFromPython(m, args);
    })
    .def_property_readonly("graph", [](Method& m) {
      return m.graph();
    })
    .def("propagate_shapes", &Method::propagate_shapes)
    .def("propagate_and_assign_input_and_output_shapes", &Method::propagate_and_assign_input_and_output_shapes)
    .def("params", &Method::params)
    .def("graph_for", [](Method& self, py::args args) {
      return self.graph_for(createVariableTensorList(args));
    });

  m.def("_jit_script_compile", [](Def def, ResolutionCallback rcb) {
    return compileFunction(def, pythonResolver(rcb));
  });

}

} // namespace script
} // namespace jit
} // namespace torch
