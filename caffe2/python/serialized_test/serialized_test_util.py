from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from caffe2.proto import caffe2_pb2
from caffe2.python import gradient_checker
import caffe2.python.hypothesis_test_util as hu
from hypothesis import given, seed, settings
import inspect
import numpy
import os
import re
import shutil
import sys
import threading

operator_test_type = 'operator_test'
TOP_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_SUFFIX = 'data'
DATA_DIR = os.path.join(TOP_DIR, DATA_SUFFIX)
_output_context = threading.local()


def given_and_seeded(*given_args, **given_kwargs):
    def wrapper(f):
        hyp_func = given(*given_args, **given_kwargs)(f)
        fixed_seed_func = seed(0)(settings(max_examples=1)(given(
            *given_args, **given_kwargs)(f)))

        def func(self, *args, **kwargs):
            self.should_serialize = True
            fixed_seed_func(self, *args, **kwargs)
            self.should_serialize = False
            hyp_func(self, *args, **kwargs)
        return func
    return wrapper


class SerializedTestCase(hu.HypothesisTestCase):

    should_serialize = False

    def get_output_dir(self):
        class_path = inspect.getfile(self.__class__)
        file_name_components = os.path.basename(class_path).split('.')
        test_file = file_name_components[0]

        function_name_components = self.id().split('.')
        test_function = function_name_components[-1]

        output_dir_arg = getattr(_output_context, 'output_dir', DATA_DIR)
        output_dir = os.path.join(
            output_dir_arg, operator_test_type, test_file + '.' + test_function)

        if os.path.exists(output_dir):
            return output_dir

        # fall back to pwd
        cwd = os.getcwd()
        serialized_util_module_components = __name__.split('.')
        serialized_util_module_components.pop()
        serialized_dir = '/'.join(serialized_util_module_components)
        output_dir_fallback = os.path.join(cwd, serialized_dir, DATA_SUFFIX)
        output_dir = os.path.join(
            output_dir_fallback,
            operator_test_type,
            test_file + '.' + test_function)

        return output_dir

    def serialize_test(self, inputs, outputs, grad_ops, op, device_option):
        def prepare_dir(path):
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        output_dir = self.get_output_dir()
        prepare_dir(output_dir)
        for (i, grad) in enumerate(grad_ops):
            grad_path = os.path.join(output_dir, 'gradient_{}.pb'.format(i))
            with open(grad_path, 'wb') as f:
                f.write(grad.SerializeToString())
        device_type = int(device_option.device_type)
        op_path = os.path.join(output_dir, 'operator_{}.pb'.format(device_type))
        with open(op_path, 'wb') as f:
            f.write(op.SerializeToString())
        numpy.savez_compressed(
            os.path.join(output_dir, 'inputs'), inputs=inputs)
        numpy.savez_compressed(
            os.path.join(output_dir, 'outputs'), outputs=outputs)

    def compare_test(self, inputs, outputs, grad_ops, atol=1e-7, rtol=1e-7):

        def parse_proto(x):
            proto = caffe2_pb2.OperatorDef()
            proto.ParseFromString(x)
            return proto

        source_dir = self.get_output_dir()

        # load serialized input and output
        loaded_inputs = numpy.load(
            os.path.join(source_dir, 'inputs.npz'), encoding='bytes')['inputs']
        inputs_equal = True
        for (x, y) in zip(inputs, loaded_inputs):
            if not numpy.array_equal(x, y):
                inputs_equal = False
        loaded_outputs = numpy.load(os.path.join(
            source_dir, 'outputs.npz'), encoding='bytes')['outputs']

        # load operator
        found_op = False
        for i in os.listdir(source_dir):
            op_file = os.path.join(source_dir, i)
            match = re.search('operator_(.+?)\.pb', i)
            if os.path.isfile(op_file) and match:
                with open(op_file, 'rb') as f:
                    loaded_op = f.read()
                op_proto = parse_proto(loaded_op)
                device_type = int(match.group(1))
                device_option = caffe2_pb2.DeviceOption(device_type=device_type)
                grad_ops, _ = gradient_checker.getGradientForOp(op_proto)
                found_op = True
                break

        # if inputs are not the same, run serialized input through serialized op
        if not inputs_equal:
            self.assertTrue(found_op)
            outputs = hu.runOpOnInput(device_option, op_proto, loaded_inputs)

        # assert outputs are equal
        for (x, y) in zip(outputs, loaded_outputs):
            numpy.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

        # assert gradient op is equal
        for i in range(len(grad_ops)):
            with open(os.path.join(source_dir, 'gradient_{}.pb'.format(i)), 'rb') as f:
                loaded_grad = f.read()
            grad_proto = parse_proto(loaded_grad)
            self.assertTrue(grad_proto == grad_ops[i])

    def assertSerializedOperatorChecks(
            self,
            inputs,
            outputs,
            gradient_operator,
            op,
            device_option,
    ):
        if self.should_serialize:
            if getattr(_output_context, 'should_write_output', False):
                self.serialize_test(
                    inputs, outputs, gradient_operator, op, device_option)
            else:
                self.compare_test(inputs, outputs, gradient_operator)

    def assertReferenceChecks(
        self,
        device_option,
        op,
        inputs,
        reference,
        input_device_options=None,
        threshold=1e-4,
        output_to_grad=None,
        grad_reference=None,
        atol=None,
        outputs_to_check=None,
    ):
        outs = super(SerializedTestCase, self).assertReferenceChecks(
            device_option,
            op,
            inputs,
            reference,
            input_device_options,
            threshold,
            output_to_grad,
            grad_reference,
            atol,
            outputs_to_check,
        )
        grad_ops, _ = gradient_checker.getGradientForOp(op)
        self.assertSerializedOperatorChecks(
            inputs,
            outs,
            grad_ops,
            op,
            device_option,
        )


def testWithArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--generate-serialized', action='store_true', dest='write',
        help='generate output files (default=false, compares to current files)')
    parser.add_argument(
        '-o', '--output', default=DATA_DIR,
        help='output directory (default: %(default)s)')
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args
    _output_context.__setattr__('should_write_output', args.write)
    _output_context.__setattr__('output_dir', args.output)

    import unittest
    unittest.main()
