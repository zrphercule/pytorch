import os
import unittest
import sys

import torch
import torch.utils.cpp_extension
import torch.backends.cudnn
try:
    import torch_test_cpp_extension.cpp as cpp_extension
except ImportError:
    print("\'test_cpp_extensions.py\' cannot be invoked directly. " +
          "Run \'python run_test.py -i cpp_extensions\' for the \'test_cpp_extensions.py\' tests.")
    raise

import common

from torch.utils.cpp_extension import CUDA_HOME
TEST_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
TEST_CUDNN = False
if TEST_CUDA:
    CUDNN_HEADER_EXISTS = os.path.isfile(os.path.join(CUDA_HOME, 'include/cudnn.h'))
    TEST_CUDNN = TEST_CUDA and CUDNN_HEADER_EXISTS and torch.backends.cudnn.is_available()


class TestCppExtension(common.TestCase):
    def test_extension_function(self):
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = cpp_extension.sigmoid_add(x, y)
        self.assertEqual(z, x.sigmoid() + y.sigmoid())

    def test_extension_module(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4)
        expected = mm.get().mm(weights)
        result = mm.forward(weights)
        self.assertEqual(expected, result)

    def test_backward(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, requires_grad=True)
        result = mm.forward(weights)
        result.sum().backward()
        tensor = mm.get()

        expected_weights_grad = tensor.t().mm(torch.ones([4, 4]))
        self.assertEqual(weights.grad, expected_weights_grad)

        expected_tensor_grad = torch.ones([4, 4]).mm(weights.t())
        self.assertEqual(tensor.grad, expected_tensor_grad)

    def test_jit_compile_extension(self):
        module = torch.utils.cpp_extension.load(
            name='jit_extension',
            sources=[
                'cpp_extensions/jit_extension.cpp',
                'cpp_extensions/jit_extension2.cpp'
            ],
            extra_include_paths=['cpp_extensions'],
            extra_cflags=['-g'],
            verbose=True)
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

        # Checking we can call a method defined not in the main C++ file.
        z = module.exp_add(x, y)
        self.assertEqual(z, x.exp() + y.exp())

        # Checking we can use this JIT-compiled class.
        doubler = module.Doubler(2, 2)
        self.assertIsNone(doubler.get().grad)
        self.assertEqual(doubler.get().sum(), 4)
        self.assertEqual(doubler.forward().sum(), 8)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cuda_extension(self):
        import torch_test_cpp_extension.cuda as cuda_extension

        x = torch.zeros(100, device='cuda', dtype=torch.float32)
        y = torch.zeros(100, device='cuda', dtype=torch.float32)

        z = cuda_extension.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_jit_cuda_extension(self):
        # NOTE: The name of the extension must equal the name of the module.
        module = torch.utils.cpp_extension.load(
            name='torch_test_cuda_extension',
            sources=[
                'cpp_extensions/cuda_extension.cpp',
                'cpp_extensions/cuda_extension.cu'
            ],
            extra_cuda_cflags=['-O2'],
            verbose=True)

        x = torch.zeros(100, device='cuda', dtype=torch.float32)
        y = torch.zeros(100, device='cuda', dtype=torch.float32)

        z = module.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not TEST_CUDNN, "CuDNN not found")
    def test_jit_cudnn_extension(self):
        # implementation of CuDNN ReLU
        if sys.platform == 'win32':
            extra_ldflags = ['cudnn.lib']
        else:
            extra_ldflags = ['-lcudnn']
        module = torch.utils.cpp_extension.load(
            name='torch_test_cudnn_extension',
            sources=[
                'cpp_extensions/cudnn_extension.cpp'
            ],
            extra_ldflags=extra_ldflags,
            verbose=True,
            with_cuda=True)

        x = torch.randn(100, device='cuda', dtype=torch.float32)
        y = torch.zeros(100, device='cuda', dtype=torch.float32)
        module.cudnn_relu(x, y)  # y=relu(x)
        self.assertEqual(torch.nn.functional.relu(x), y)
        with self.assertRaisesRegex(RuntimeError, "same size"):
            y_incorrect = torch.zeros(20, device='cuda', dtype=torch.float32)
            module.cudnn_relu(x, y_incorrect)

    def test_optional(self):
        has_value = cpp_extension.function_taking_optional(torch.ones(5))
        self.assertTrue(has_value)
        has_value = cpp_extension.function_taking_optional(None)
        self.assertFalse(has_value)

    def test_inline_jit_compile_extension_with_functions_as_list(self):
        cpp_source = '''
        at::Tensor tanh_add(at::Tensor x, at::Tensor y) {
          return x.tanh() + y.tanh();
        }
        '''

        module = torch.utils.cpp_extension.load_inline(
            name='inline_jit_extension_with_functions_list',
            cpp_sources=cpp_source,
            functions='tanh_add',
            verbose=True)

        self.assertEqual(module.tanh_add.__doc__.split('\n')[2], 'tanh_add')

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

    def test_inline_jit_compile_extension_with_functions_as_dict(self):
        cpp_source = '''
        at::Tensor tanh_add(at::Tensor x, at::Tensor y) {
          return x.tanh() + y.tanh();
        }
        '''

        module = torch.utils.cpp_extension.load_inline(
            name='inline_jit_extension_with_functions_dict',
            cpp_sources=cpp_source,
            functions={'tanh_add': 'Tanh and then sum :D'},
            verbose=True)

        self.assertEqual(
            module.tanh_add.__doc__.split('\n')[2], 'Tanh and then sum :D')

    def test_inline_jit_compile_extension_multiple_sources_and_no_functions(self):
        cpp_source1 = '''
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        '''

        cpp_source2 = '''
        #include <torch/torch.h>
        at::Tensor sin_add(at::Tensor x, at::Tensor y);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
          m.def("sin_add", &sin_add, "sin(x) + sin(y)");
        }
        '''

        module = torch.utils.cpp_extension.load_inline(
            name='inline_jit_extension',
            cpp_sources=[cpp_source1, cpp_source2],
            verbose=True)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.sin_add(x, y)
        self.assertEqual(z, x.sin() + y.sin())

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_inline_jit_compile_extension_cuda(self):
        cuda_source = '''
        __global__ void cos_add_kernel(
            const float* __restrict__ x,
            const float* __restrict__ y,
            float* __restrict__ output,
            const int size) {
          const auto index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index < size) {
            output[index] = __cosf(x[index]) + __cosf(y[index]);
          }
        }

        at::Tensor cos_add(at::Tensor x, at::Tensor y) {
          auto output = at::zeros_like(x);
          const int threads = 1024;
          const int blocks = (output.numel() + threads - 1) / threads;
          cos_add_kernel<<<blocks, threads>>>(x.data<float>(), y.data<float>(), output.data<float>(), output.numel());
          return output;
        }
        '''

        # Here, the C++ source need only declare the function signature.
        cpp_source = 'at::Tensor cos_add(at::Tensor x, at::Tensor y);'

        module = torch.utils.cpp_extension.load_inline(
            name='inline_jit_extension_cuda',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['cos_add'],
            verbose=True)

        self.assertEqual(module.cos_add.__doc__.split('\n')[2], 'cos_add')

        x = torch.randn(4, 4, device='cuda', dtype=torch.float32)
        y = torch.randn(4, 4, device='cuda', dtype=torch.float32)

        z = module.cos_add(x, y)
        self.assertEqual(z, x.cos() + y.cos())

    def test_inline_jit_compile_extension_throws_when_functions_is_bad(self):
        with self.assertRaises(ValueError):
            torch.utils.cpp_extension.load_inline(
                name='invalid_jit_extension', cpp_sources='', functions=5)

    def test_lenient_flag_handling_in_jit_extensions(self):
        cpp_source = '''
        at::Tensor tanh_add(at::Tensor x, at::Tensor y) {
          return x.tanh() + y.tanh();
        }
        '''

        module = torch.utils.cpp_extension.load_inline(
            name='lenient_flag_handling_extension',
            cpp_sources=cpp_source,
            functions='tanh_add',
            extra_cflags=['-g\n\n', '-O0 -Wall'],
            extra_include_paths=['       cpp_extensions\n', '../'],
            verbose=True)

        x = torch.zeros(100, dtype=torch.float32)
        y = torch.zeros(100, dtype=torch.float32)
        z = module.tanh_add(x, y).cpu()
        self.assertEqual(z, x.tanh() + y.tanh())

    def test_complex_registration(self):
        module = torch.utils.cpp_extension.load(
            name='complex_registration_extension',
            sources='cpp_extensions/complex_registration_extension.cpp',
            verbose=True)

        torch.empty(2, 2, dtype=torch.complex64)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_half_support(self):
        '''
        Checks for an issue with operator< ambiguity for half when certain
        THC headers are included.

        See https://github.com/pytorch/pytorch/pull/10301#issuecomment-416773333
        for the corresponding issue.
        '''
        cuda_source = '''
        #include <THC/THCNumerics.cuh>

        template<typename T, typename U>
        __global__ void half_test_kernel(const T* input, U* output) {
            if (input[0] < input[1] || input[0] >= input[1]) {
                output[0] = 123;
            }
        }

        at::Tensor half_test(at::Tensor input) {
            auto output = at::empty(1, input.options().dtype(at::kFloat));
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "half_test", [&] {
                half_test_kernel<scalar_t><<<1, 1>>>(
                    input.data<scalar_t>(),
                    output.data<float>());
            });
            return output;
        }
        '''

        module = torch.utils.cpp_extension.load_inline(
            name='half_test_extension',
            cpp_sources='at::Tensor half_test(at::Tensor input);',
            cuda_sources=cuda_source,
            functions=['half_test'],
            verbose=True)

        x = torch.randn(3, device='cuda', dtype=torch.half)
        result = module.half_test(x)
        self.assertEqual(result[0], 123)


if __name__ == '__main__':
    common.run_tests()
