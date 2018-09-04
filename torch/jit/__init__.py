import torch._C
from torch import Tensor
from torch.autograd import Variable, function
from torch.nn import Module, ModuleList, ParameterList, Parameter, Sequential
from torch.jit.frontend import get_jit_ast
import torch.jit.annotations
from torch._six import raise_from, with_metaclass
import torch.testing
from collections import defaultdict, OrderedDict, namedtuple, Iterable
import sys
import warnings
import itertools
import weakref
import types
import contextlib
import os
import functools
import inspect
import copy
import numbers
import collections
import re


def _parse_env(name, default, true_message, false_message):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {'1', 'true', 'yes'}:
        return True
    elif value.lower() in {'0', 'false', 'no'}:
        return False
    if value == '1v':
        print(true_message)
        return True
    elif value == '0v':
        print(false_message)
        return False
    raise ValueError('Unknown setting of {}. Try using 0 or 1.'.format(name))


_enabled = _parse_env('PYTORCH_JIT', True, "> Using PyTorch JIT", "> PyTorch JIT DISABLED")
_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten
_jit_script_compile = torch._C._jit_script_compile
BatchTensor = torch._C._jit.BatchTensor


@contextlib.contextmanager
def scope(scope_name):
    tracing_state = torch._C._get_tracing_state()
    if tracing_state:
        tracing_state.push_scope(scope_name)
    try:
        yield
    finally:
        if tracing_state:
            tracing_state.pop_scope()


def load(filename):
    m = ScriptModule()
    m._load(filename)
    return m


def get_trace_graph(f, args=(), kwargs=None):
    """
    Trace a function or model, returning a tuple consisting of the both the
    *trace* of an execution, as well as the original return value.

    Tracing is guaranteed not to change the semantics of the function/module
    that is traced.

    Arguments:
        f (torch.nn.Module or function): the function or module
            to be traced.
        args (tuple or Tensor): the positional arguments to pass to the
            function/module to be traced.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        kwargs (dict): the keyword arguments to pass to the function/module
            to be traced.

    Example: Trace a cell.

        >>> trace, out = jit.trace(nn.LSTMCell(), (input, hidden))
        >>> print(trace)
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    return LegacyTracedModule(f)(*args, **kwargs)


def _unique_state_dict(module, keep_vars=False):
    state_dict = module.state_dict(keep_vars=keep_vars)
    filtered_dict = type(state_dict)()
    seen_ids = set()
    for k, v in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        filtered_dict[k] = v
    return filtered_dict


class LegacyTracedModule(Module):
    def __init__(self, inner):
        super(LegacyTracedModule, self).__init__()
        # inner may be a Module, or it may be an arbitrary callable
        # If it's a Module, we get its parameters automatically, which lets
        # us avoid a special casing functions versus modules.
        self.inner = inner

    def forward(self, *args):
        in_vars, in_desc = _flatten(args)
        # NOTE: use full state, because we need it for BatchNorm export
        # This differs from the compiler path, which doesn't support it at the moment.
        module_state = list(_unique_state_dict(self, keep_vars=True).values())
        trace, all_trace_inputs = torch._C._tracer_enter(*(in_vars + module_state))
        try:
            trace_inputs = _unflatten(all_trace_inputs[:len(in_vars)], in_desc)
            out = self.inner(*trace_inputs)
            out_vars, _ = _flatten(out)
            torch._C._tracer_exit(tuple(out_vars))
        except Exception:
            torch._C._tracer_abandon()
            raise
        return trace, out


def _clone_inputs(args):
    def clone_input(a):
        if a is None:
            return None
        elif isinstance(a, torch.Tensor):
            # TODO: figure out one liner to .clone() and set requires_grad
            v = Variable(a.data.clone(), requires_grad=a.requires_grad)
            if a.grad is not None:
                v.grad = clone_input(v.grad)
            return v
        else:
            return a.clone()
    return function._nested_map(lambda x: isinstance(x, torch.Tensor),
                                clone_input, condition_msg="tensors")(args)


# This is purely for developer debugging.  We are not going to advertise it.
_JIT_DUMP = os.environ.get('PYTORCH_JIT_DUMP', False)
_JIT_TIME = os.environ.get('PYTORCH_JIT_TIME', False)  # CUDA-only timing
_JIT_DISABLE = os.environ.get('PYTORCH_JIT_DISABLE', False)
_JIT_STATS = os.environ.get('PYTORCH_JIT_STATS', False)


def _dump_trace(trace_name, pass_name, input_key, trace):
    if not _JIT_DUMP:
        return

    import torch.contrib._graph_vis as graph_vis

    filename = "{}_{}".format(trace_name, pass_name)
    # TODO: Also paste out the backtrace when the trace was compiled
    # (and maybe also when it was run?)
    with open(filename + ".ir", "w") as f:
        f.write("Input key: {}\n\n{}".format(input_key, str(trace)))
    graph_vis.write(trace.graph(), filename + ".html")


@contextlib.contextmanager
def _time(trace_name, name, time=True):
    if (not _JIT_TIME and not time) or not torch.cuda.is_available():
        yield
        return
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.record_event(start)
    try:
        yield
    finally:
        stream.record_event(end)
        end.synchronize()
        print("{} {} time: {} ms".format(trace_name, name, start.elapsed_time(end)))


def verify(model, args, loss_fn=torch.sum, devices=None):
    """
    Verify that a JIT compiled model has the same behavior as its uncompiled
    version along with its backwards pass.  If your model returns multiple
    outputs, you must also specify a `loss_fn` to produce a loss for which
    the backwards will be computed.

    This function has side-effects (e.g., it executes your model / saves and loads
    parameters), so don't expect the model to come out exactly the same as what
    you passed in.

    Arguments:
        model (compiled torch.nn.Module or function): the module/function to be
            verified.  The module/function definition MUST have been decorated with
            `@torch.jit.compile`.
        args (tuple or Tensor): the positional arguments to pass to the
            compiled function/module to be verified.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        loss_fn (function, optional): the loss function to be applied to
            the output of the model, before backwards is invoked.  By default,
            we assume that a model returns a single result, and we :func:`torch.sum`
            before calling backwards; if this is inappropriate, you can pass your
            own loss function.  Note that if a model returns a tuple of results,
            these are passed as separate positional arguments to `loss_fn`.
        devices (iterable of device IDs, optional): the GPU devices which the
            compiled module will be run on.  This determines the RNG state we
            must save when running both compiled and uncompiled versions of the model.
    """
    # TODO: In principle, we track device information in our trace, so it
    # should be possible to check if our execution actually obeyed the 'devices'
    # the user provided.

    # TODO: Consider adding a utility function to torch.jit to test
    # for this case
    if not isinstance(model, torch._C.CompiledFunction):
        raise TypeError("Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it")
    is_module = isinstance(model, Module)

    if not isinstance(args, tuple):
        args = (args,)

    saved_args = _clone_inputs(args)
    if is_module:
        saved_state = copy.deepcopy(model.state_dict())

    def run_fwd_bwd(args, force_trace=False, assert_compiled=False):
        params = list(model.parameters()) if is_module else []
        in_vars, _ = _flatten((args, params))
        # We use a special API to reset the trace and compile it from scratch.
        compiled_fn = model
        if force_trace:
            compiled_fn.clear_cache()
        if assert_compiled:
            hits = compiled_fn.hits
        out = model(*args)
        if assert_compiled and compiled_fn.hits == hits:
            raise RuntimeError("failed to use the compiled function")
        if not isinstance(out, tuple):
            out = (out, )
        if loss_fn == torch.sum and len(out) != 1:
            raise ValueError(("Model returns {} outputs, but default loss function "
                              "(torch.sum) can only handle a single output").format(len(out)))
        out_vars, _ = _flatten(out)
        saved_outs = [v.data.clone() for v in out_vars]
        loss = loss_fn(*out)
        grads = torch.autograd.grad([loss], in_vars)
        # TODO: I'm not sure if the clone here is necessary but it is safer
        saved_grads = [v.data.clone() for v in grads]
        return (saved_outs, saved_grads)

    with torch.random.fork_rng(devices, _caller="torch.jit.verify"):
        uncompiled_outs, uncompiled_grads = run_fwd_bwd(args, force_trace=True)
        assert model.has_trace_for(*args)

    if is_module:
        model.load_state_dict(saved_state)
    compiled_outs, compiled_grads = run_fwd_bwd(args, assert_compiled=True)

    _verify_equal(uncompiled_outs, compiled_outs)
    _verify_equal(uncompiled_grads, compiled_grads)


def _verify_equal(xs, ys):
    for x, y in zip(xs, ys):
        if x.sub(y).abs().max() > 1e-6:
            raise RuntimeError("JIT and real computation mismatch")


def indent(s):
    return '\n'.join(['\t' + line for line in s.splitlines()])


class TracingCheckError(Exception):
    def __init__(self, graph_diff_error, tensor_compare_error, nondeterm_warning, extra_msg=None):
        self.message = 'Tracing failed sanity checks!\n'
        if extra_msg is not None:
            self.message += extra_msg + '\n'
        if graph_diff_error is not None:
            self.message += 'ERROR: Graphs differed across invocations!\n'
            self.message += indent(graph_diff_error) + '\n'
        if nondeterm_warning is not None:
            self.message += 'WARNING: '
            self.message += nondeterm_warning + '\n'
        if tensor_compare_error is not None:
            self.message += 'ERROR: Tensor-valued Constant nodes differed in value ' \
                            'across invocations. This often indicates that the tracer has' \
                            ' encountered untraceable code.\n'
            self.message += indent(tensor_compare_error) + '\n'
        super(TracingCheckError, self).__init__(self.message)


# Check the traced module against a set of user-provided validation inputs
def _check_trace(check_inputs, func, executor_options, module, check_tolerance):
    for inputs in check_inputs:
        check_mod = torch.jit.trace(func, _clone_inputs(inputs), check_trace=False, **executor_options)

        def graph_diagnostic_info():
            mod_canonicalized = torch._C._jit_pass_canonicalize(module.graph)
            torch._C._jit_pass_erase_shape_information(mod_canonicalized)
            check_canonicalized = torch._C._jit_pass_canonicalize(check_mod.graph)
            torch._C._jit_pass_erase_shape_information(check_canonicalized)

            graph_diff_errors = None
            if str(mod_canonicalized) != str(check_canonicalized):
                import difflib
                graph_diff = difflib.ndiff(str(mod_canonicalized).splitlines(True),
                                           str(check_canonicalized).splitlines(True))
                graph_diff_errors = 'Graph diff:\n' + indent(''.join(graph_diff)) + '\n'

                for n_mod, n_check in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
                    if str(n_mod) != str(n_check):
                        graph_diff_errors += 'First diverging operator:\n'
                        node_diff = difflib.ndiff(str(n_mod).splitlines(True),
                                                  str(n_check).splitlines(True))
                        source_printout = 'Node diff:\n' + indent(''.join(node_diff)) + '\n'
                        mod_stack = n_mod.getSourceLocation()
                        if mod_stack:
                            source_printout += 'Trace source location:\n' + indent(mod_stack) + '\n'
                        check_stack = n_check.getSourceLocation()
                        if check_stack:
                            source_printout += 'Check source location:\n' + indent(check_stack) + '\n'
                        graph_diff_errors += source_printout

                        break  # For now, only print out the first pair of nodes that diverges

            tensor_compare_errors = None
            # Check Tensor-valued constant nodes
            for n_mod, n_check in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
                if n_mod.kind() != n_check.kind():
                    break  # Graphs have already diverged

                if n_mod.kind() == n_check.kind() and n_mod.kind() == 'prim::Constant':
                    if n_mod.kindOf('value') != 't' or n_check.kindOf('value') != 't':
                        continue

                    mod_tensor_val = n_mod.t('value')
                    check_tensor_val = n_check.t('value')

                    try:
                        torch.testing.assert_allclose(mod_tensor_val, check_tensor_val)
                    except (RuntimeError, AssertionError) as e:
                        if not tensor_compare_errors:
                            tensor_compare_errors = ''
                        tensor_compare_errors += 'Node:\n' + indent(str(n_mod)) + '\n'
                        compare_stack = n_mod.getSourceLocation()
                        if compare_stack:
                            tensor_compare_errors += 'Source Location:\n' + indent(compare_stack) + '\n'
                        tensor_compare_errors += 'Comparison exception: ' + indent(str(e))

                        break  # For now, only print the first diverging pair

            nondeterministic_ops_warning = None
            nondeterm_ops = [op for op in module.graph.nodes() if op.isNondeterministic()]
            if len(nondeterm_ops) > 0:
                nondeterministic_ops_warning = "Trace had nondeterministic nodes. Nodes:\n"
                for op in nondeterm_ops:
                    nondeterministic_ops_warning += indent(str(op))
                nondeterministic_ops_warning += "\nThis may cause errors in trace checking. To disable trace checking,"\
                                                " pass disable_checks=True to torch.jit.trace()"

            return graph_diff_errors, tensor_compare_errors, nondeterministic_ops_warning

        def wrap_retval(x):
            return x if isinstance(x, tuple) else (x,)

        def run_mod_and_filter_tensor_outputs(mod, inputs):
            outs = wrap_retval(mod(*_clone_inputs(inputs)))
            outs = [out for out in outs if isinstance(out, torch.Tensor)]
            return outs

        try:
            traced_outs = run_mod_and_filter_tensor_outputs(module, inputs)
        except Exception as e:
            msg = 'Encountered an exception while running trace with check inputs.\nException:\n' + indent(str(e))
            raise TracingCheckError(*graph_diagnostic_info(), extra_msg=msg)

        try:
            fn_outs = run_mod_and_filter_tensor_outputs(func, inputs)
            for orig, check in zip(traced_outs, fn_outs):
                torch.testing.assert_allclose(orig.double(), check.double(), rtol=check_tolerance,
                                              atol=torch.testing._get_default_tolerance(orig, check)[1])
        except (RuntimeError, AssertionError) as e:
            # TODO: interpose on tracing the function again and check for
            # divergence? then we can point to where in the source code
            # we start diverging in python v.s. the trace
            msg = 'ERROR: Traced function outputs do not match the Python function outputs.\nException: ' + str(e)
            raise TracingCheckError(*graph_diagnostic_info(),
                                    extra_msg=msg)

        try:
            check_outs = run_mod_and_filter_tensor_outputs(check_mod, inputs)
        except Exception as e:
            msg = 'Encountered an exception while running checking trace with check inputs.\nException:\n' \
                + indent(str(e))
            raise TracingCheckError(*graph_diagnostic_info(), extra_msg=msg)

        try:
            for orig, check in zip(traced_outs, check_outs):
                torch.testing.assert_allclose(orig.double(), check.double(), rtol=check_tolerance,
                                              atol=torch.testing._get_default_tolerance(orig, check)[1])
        except (RuntimeError, AssertionError) as e:
            raise TracingCheckError(*graph_diagnostic_info())


class TracerWarning(Warning):
    @staticmethod
    def ignore_lib_warnings():
        warnings.filterwarnings('ignore', category=TracerWarning, module='torch.*')


# We ignore the tracer warnings coming form inside the library, because all our shape
# checks in nn will trigger them.
TracerWarning.ignore_lib_warnings()
torch._C._tracer_warn_use_python()


def trace(func, example_inputs, optimize=True, check_trace=True, check_inputs=None, check_tolerance=1e-5):
    """
    Trace a function and return an executable trace that will be optimized
    using just-in-time compilation.

    .. warning::

        Just-in-time compilation currently only works for functions/modules
        which are not data dependent (e.g., have conditionals on data in
        tensors) and do not have any untracked external dependencies (e.g.,
        perform input/output or access global variables). If you trace such
        models, you will silently get incorrect results on subsequent
        invocations of the model.

    Arg:
        func - a python function or torch.nn.Module that will be run with example_inputs.
               arguments and returns to func must be Tensors or (possibly nested) tuples that
               contain tensors.
        example_inputs - a tuple of example inputs that will be passed to the function
                         while tracing. The resulting trace can be run with
                         inputs of different types and shapes assuming the traced operations
                         support those types and shapes. example_inputs may also be a single
                         Tensor in which case it is automatically wrapped in a tuple

    Keyword arguments:
        optimize (bool, optional): whether or not to apply optimizations.  Default: ``True``.
        check_trace (bool, optional): check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of tuples. optional): A list of tuples of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a seet of input arguments that would
                                                 be specified in `args`. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original `args` is used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.

    Returns:
        A torch.jit.ScriptModule object with a single forward() method containing the traced code.
        When func in s a torch.nn.Module, the returned ScriptModule will have the same set of
        sub-modules and parameters as func.

    Example:
       >>> def f(x):
       ...     return x * 2
       >>> traced_f = torch.jit.trace(f, torch.rand(1))

    """
    if not _enabled:
        return func
    executor_options = {'optimize': bool(optimize)}
    # Special case for common case of passing a single Tensor
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    elif not isinstance(example_inputs, tuple):
        example_inputs = tuple(example_inputs)
    module = TopLevelTracedModule(func, **executor_options)
    module._create_method_from_trace('forward', func, example_inputs)

    # Check the trace against new traces created from user-specified inputs
    if check_trace:
        if check_inputs is not None:
            _check_trace(check_inputs, func, executor_options, module, check_tolerance)
        else:
            _check_trace([example_inputs], func, executor_options, module, check_tolerance)

    return module


def createResolutionCallback(frames_up=0):
    """
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallback (by default).
    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallback()
            print(x("foo"))

        def baz():
            foo = 2
            bar()

        baz()

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is
    """
    frame = inspect.stack()[1 + frames_up][0]

    def env(key):
        if key in frame.f_locals:
            return frame.f_locals[key]
        elif key in frame.f_globals:
            return frame.f_globals[key]
        else:
            return None

    return env


class CompilationUnit(object):
    def __init__(self, lang=None, optimize=True, _frames_up=0):
        self.module = torch._C.ScriptModule()
        self.module._set_optimized(optimize)
        if lang is not None:
            self.define(lang, _frames_up=_frames_up + 1)
        self.optimize = optimize

    def define(self, lang, rcb=None, _frames_up=0):
        if not rcb:
            rcb = createResolutionCallback(_frames_up + 1)
        self.module._define(lang, rcb, False)

    def __getattr__(self, attr):
        return self.module._get_method(attr)


def script(fn, optimize=True, _frames_up=0):
    if not _enabled:
        return fn
    rcb = createResolutionCallback(_frames_up + 1)
    ast = get_jit_ast(fn, is_method=False)
    graph = _jit_script_compile(ast, rcb)
    mod = ScriptModule()
    mod._create_method_from_graph('forward', graph)
    # TODO: refactor everything so we're not 1) creating a ScriptModule
    # 2) Throwing everything away except for the graph 3) Creating a new
    # ScriptModule and dumping that graph in 4) Re-populating the schema
    # because it was lost doing the previous
    mod.__getattr__('forward').forward_schema(ast, False)
    # Forward docstrings
    mod.__doc__ = fn.__doc__
    return mod


ScriptMethodStub = namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))


def script_method(fn):
    if not _enabled:
        return fn
    # NOTE: we need to traverse two frames here because the meta-class frame
    # for ScriptModule will be present, as opposed to invoking @script on a
    # a function or invoking define() on a CompilationUnit.
    # The stack will look like:
    #
    # 0. createResolutionCallback()
    # 1. script_method()
    # 2. ScriptModule metaclass frame
    # 3. Surrounding scope
    #
    # createResolutionCallback internally adds 1 to get us to the scope of this
    # function (the calling function). Adding 2 gets us to the proper surrounding scope.
    rcb = createResolutionCallback(frames_up=2)
    ast = get_jit_ast(fn, is_method=True)
    return ScriptMethodStub(rcb, ast, fn)


def batch(batch_size=1, optimize=True, _frames_up=0):
    def decorator(fn):
        if not _enabled:
            return fn
        import torch.jit.batchop
        mod = script(fn, optimize, _frames_up)
        res_graph = torch.to_batch_graph(mod.graph)
        res_mod = ScriptModule()
        res_mod._create_method_from_graph('forward', res_graph)

        def wrapper(*args):
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg = BatchTensor(arg, batch_size)
                if isinstance(arg, BatchTensor):
                    new_args.extend([arg.get_data(), arg.get_mask(), arg.get_dims()])
                else:
                    new_args.append(arg)
            res = res_mod(*new_args)
            assert len(res) % 3 == 0
            if len(res) % 3 != 0:
                raise "non-batched-tensor output is not supported yet"
            result = [BatchTensor(*res[i * 3: i * 3 + 3]) for i in range(len(res) // 3)]
            if len(result) == 1:
                return result[0]
            return result
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


# These OrderedDictWrapper classes replace the actual OrderedDicts in
# module with versions that get/set properties inside of script::Module.
# This allows us to reuse most of nn.Module while still storing the
# data in C++.
# Each OrderedDict needs to support:
#  x not in view
#  x in view
#  view[name] = ...
#  view.values()
#  del view[name]
#  view.items()
#  view.keys()
#  len(view)

class OrderedDictWrapper(object):
    def __init__(self, module):
        self.module_ref = weakref.ref(module)

    @property
    def module(self):
        r = self.module_ref()
        if r is None:
            raise RuntimeError("_parameters or _modules alive after module is dead")
        return r

    def keys(self):
        return [k for k, v in self.items()]

    def values(self):
        return [v for k, v in self.items()]

    def __delitem__(self, k):
        raise RuntimeError("cannot delete methods or parameters of a script module")

    def items(self):
        raise NotImplementedError

    def __contains__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        raise NotImplementedError

    def __setitem__(self, k, v):
        raise NotImplementedError


class OrderedModuleDict(OrderedDictWrapper):
    def __init__(self, module):
        super(OrderedModuleDict, self).__init__(module)
        # contains _both_ script modules and non-script python-only modules

        # because script modules are subclassed in python and the
        # C++ script::Module class will not hold references to them,
        # to ensure that you always get the same python value here
        # we store it in the python dict as well
        self._python_modules = OrderedDict()

    def items(self):
        r = self._python_modules.items()
        return r

    def __contains__(self, k):
        return k in self._python_modules

    def __setitem__(self, k, v):
        if k in self._python_modules:
            raise RuntimeError("cannot re-assign modules in a ScriptModule")
        if isinstance(v, ScriptModule):
            self.module._register_module(k, v)

        self._python_modules[k] = v

    def __getitem__(self, k):
        return self._python_modules[k]


class OrderedParameterDict(OrderedDictWrapper):
    def __init__(self, module):
        super(OrderedParameterDict, self).__init__(module)

    def items(self):
        return [(name, param) for name, param, is_buffer
                in self.module._get_parameters()
                if not is_buffer]

    def __setitem__(self, k, v):
        self.module._register_parameter(k, v, False)

    def __contains__(self, k):
        return self.module._has_parameter(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self.module._get_parameter(k)


class OrderedBufferDict(OrderedDictWrapper):
    def __init__(self, module):
        super(OrderedBufferDict, self).__init__(module)

    def items(self):
        return [(name, param) for name, param, is_buffer
                in self.module._get_parameters()
                if is_buffer]

    def __setitem__(self, k, v):
        self.module._register_parameter(k, v, True)

    def __contains__(self, k):
        return self.module._has_buffer(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self.module._get_parameter(k)

# base types that can be constants
# in addition, tuples and lists of these base types are also considered constants
# If you edit this list, then you also need to edit the handlers in
# ConstantValue in jit/script/init.cpp
_constant_types = (bool, float, int, types.FunctionType, torch.device, torch.layout, torch.dtype)


def _get_valid_constant(v):
    if isinstance(v, _constant_types):
        return v
    elif isinstance(v, tuple) or isinstance(v, list):
        return tuple(_get_valid_constant(x) for x in v)
    constants = ", ".join(typ.__name__ for typ in _constant_types)
    raise TypeError(
        "'{}' object is not a valid constant.\n".format(type(v).__name__) +
        "Valid constants are:\n" +
        "  1. a nn.ModuleList\n" +
        "  2. a value of type {{{}}}\n".format(constants) +
        "  3. a list or tuple of (2)\n")

# For each user-defined class that subclasses ScriptModule this meta-class,
# (1) finds all the methods annotated with @script_method
# in a ScriptModule and removes them from the class attributes, and
# (2) puts a wrapper around the class's __init__ method to register
# all of the script_methods with the module after the original __init__
# has run. This has to occur after the user-defined __init__ so that
# submodules and parameters are initialized _before_ the script compiler
# resolve references to `self.param` or `self.module`.


class ScriptMeta(type(torch._C.ScriptModule)):
    # this has to inherit from pybind11's metaclass otherwise we get
    # issues because ScriptModule inherits from torch._C.ScriptModule,
    # a pybind11 type
    def __init__(cls, name, bases, attrs):
        # find all the script methods
        cls._original_methods = {}
        methods = []
        for k, v in sorted(attrs.items()):
            if isinstance(v, ScriptMethodStub):
                delattr(cls, k)
                methods.append(v)
                cls._original_methods[v.original_method.__name__] = v.original_method
        # after the user's __init__ register all the script methods
        # with the module
        original_init = getattr(cls, '__init__', lambda self: None)
        super_constants = getattr(super(cls), '_constants_set', set())
        cls._constants_set = set(getattr(cls, '__constants__', ())).union(super_constants)

        def init_then_register(self, *args, **kwargs):
            # ensure even if the user forgets to call super that
            # the pybind object is initialized so it will not segfault
            # run this once, before the most-derived __init__ is called
            if cls is type(self):
                torch._C.ScriptModule.__init__(self)
            original_init(self, *args, **kwargs)
            defs = [m.def_ for m in methods]
            rcbs = [m.resolution_callback for m in methods]
            self._create_methods(defs, rcbs)

        cls.__init__ = init_then_register
        return super(ScriptMeta, cls).__init__(name, bases, attrs)


if _enabled:
    class ScriptModule(with_metaclass(ScriptMeta, torch._C.ScriptModule, Module)):
        def __init__(self, optimize=True):
            # must be before Module.init since the field is used in __getattr__
            Module.__init__(self)
            self._set_optimized(optimize)
            self._parameters = OrderedParameterDict(self)
            self._buffers = OrderedBufferDict(self)
            self._modules = OrderedModuleDict(self)

        def __getattr__(self, attr):
            if self._has_method(attr):
                if attr in self.__class__._original_methods:
                    original_method = self.__class__._original_methods[attr]
                    script_method = self._get_method(attr)
                    return functools.wraps(original_method)(script_method)
                else:
                    return self._get_method(attr)
            if attr == 'graph' and self._has_method('forward'):
                return self.__getattr__('forward').graph
            return Module.__getattr__(self, attr)

        def __setattr__(self, attr, value):
            if attr not in self._constants_set:
                return super(ScriptModule, self).__setattr__(attr, value)
            if hasattr(self, attr):
                raise RuntimeError("attempting to re-assign constant '{}'".format(attr))
            if isinstance(value, ModuleList):
                # special case for list of modules. Modules need to be registered with their
                # parent module. To do this, we create a ConstModuleList, which is itself a module, that
                # contains each of these modules as submodules. The ConstModuleList then
                # is set as an attribute of the parent module.
                super(ScriptModule, self).__setattr__(attr, _ConstModuleList(value))
            elif isinstance(value, Sequential):
                super(ScriptModule, self).__setattr__(attr, _ConstSequential(value))
            else:
                super(ScriptModule, self).__setattr__(attr, _get_valid_constant(value))

        def __dir__(self):
            return sorted(Module.__dir__(self) + self._method_names())

        def define(self, lang):
            # We use frames_up=1 to get to the proper surrounding scope. The stack
            # will look like:
            # 0. createResolutionCallback
            # 1. define()
            # 2. surrounding scope.
            #
            # createResolutionCallback internally adds 1 to get us to our frame, then
            # we add 1 to get to the proper surrounding scope.
            rcb = createResolutionCallback(frames_up=1)
            self._define(lang, rcb, True)
else:
    ScriptModule = torch.nn.Module


def _get_methods(cls):
    import inspect
    # In Python 3 unbound methods are functions, but in Python 2 they are methods
    return inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))


_compiled_methods_whitelist = {
    'forward', 'register_buffer', 'register_parameter', 'add_module',
    '_apply', 'apply', 'cuda', 'cpu', 'type', 'float', 'double', 'half',
    'state_dict', 'load_state_dict', '_load_from_state_dict',
    '_named_members', 'parameters', 'named_parameters',
    'buffers', 'named_buffers', 'children', 'named_children', 'modules',
    'named_modules', 'zero_grad', 'share_memory', '_get_name', 'extra_repr',
    '_slow_forward', '_tracing_name'
}


def _make_fail(name):
    def fail(self, *args, **kwargs):
        raise RuntimeError(name + " is not supported on TracedModules")
    return fail


for name, method in _get_methods(torch.nn.Module):
    if name.startswith('__'):
        continue
    if name not in ScriptModule.__dict__ and name not in _compiled_methods_whitelist:
        setattr(ScriptModule, method.__name__, _make_fail(name))


class TracedModule(ScriptModule):
    __frozen = False

    def __init__(self, orig, id_set=None, optimize=True):
        # XXX: orig can be a nn.Module or a function!
        super(TracedModule, self).__init__(optimize=optimize)
        if id_set is None:
            id_set = set()

        if not isinstance(orig, torch.nn.Module):
            self._name = orig.__name__
            orig = torch.nn.Module()
        else:
            self._name = 'TracedModule[' + type(orig).__name__ + ']'

        def check_unique(param):
            if param in id_set:
                raise ValueError("TracedModules don't support parameter sharing between modules")
            id_set.add(param)

        self.training = orig.training

        for name, param in orig._parameters.items():
            if param is not None:
                self._parameters[name] = param
                check_unique(param)
        for name, buf in orig._buffers.items():
            if buf is not None:
                self._buffers[name] = buf
                check_unique(buf)

        if orig._backward_hooks or orig._forward_hooks or orig._forward_pre_hooks:
            raise ValueError("Modules that have hooks assigned can't be compiled")

        for name, submodule in orig._modules.items():
            self._modules[name] = TracedModule(submodule, id_set, optimize=optimize)

        self._freeze()

    def forward(self, *args, **kwargs):
        raise RuntimeError('Trace submodules cannot be called.')

    def _freeze(self):
        self.__frozen = True

    def _get_name(self):
        return self._name

    def __setattr__(self, attr, value):
        if not self.__frozen or hasattr(self, attr):
            return super(TracedModule, self).__setattr__(attr, value)
        raise RuntimeError("Cannot set new properties on a traced module.")


class TopLevelTracedModule(TracedModule):
    def forward(self, *args, **kwargs):
        return self._get_method('forward')(*args, **kwargs)


class _ConstModuleList(ScriptModule):
    def __init__(self, modules):
        super(_ConstModuleList, self).__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _ConstModuleList(list(self._modules.values())[idx])
        else:
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range'.format(idx))
            if idx < 0:
                idx += len(self)
            return self._modules[str(idx)]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __dir__(self):
        keys = super(_ConstModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys


class _ConstSequential(_ConstModuleList):
    __constants__ = ['mods']

    def __init__(self, mods):
        super(_ConstSequential, self).__init__(mods._modules.values())

        # we define the forward method via self.define rather than
        # making it a direct class member (with a @script) annotation
        # because, in optimized runtime environments where only .pyc files
        # are shipped, we cant retrieve the source code.
        # TODO: find a workaround for this and remove this hack
        self.define("""
        def forward(self, input):
            for m in self:
                input = m(input)
            return input
        """)


_builtin_table = None


# lazily built to ensure the correct initialization order
def _get_builtin_table():
    global _builtin_table
    if _builtin_table is not None:
        return _builtin_table
    _builtin_table = {}

    def register_all(mod):
        for name in dir(mod):
            v = getattr(mod, name)
            if callable(v):
                _builtin_table[id(v)] = "aten::" + name
    register_all(torch)
    register_all(torch.nn.functional)
    return _builtin_table


def _register_builtin(fn, op):
    _get_builtin_table()[id(fn)] = op


def _find_builtin(fn):
    return _get_builtin_table().get(id(fn))


class _disable_tracing(object):
    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
