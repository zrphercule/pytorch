#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/native/LinearAlgebraUtils.h"
#include <functional>
#include <numeric>
#include <vector>

namespace at {
namespace native {

// Helper function for det methods.
// For pivoted LU factorization A = P * L * U. Since we always have det(L) = 1,
// det(P) = \pm 1, this method returns a 3-tuple:
//   (det(P), diag(U), info),
// where info helps us identify singular matrices.
static inline std::tuple<double, Tensor, int> _lu_det_P_diag_U_info(const Tensor& self) {
  Tensor p, lu, info;
  std::tie(lu, p, info) = self.unsqueeze(0).btrifact_with_info();
  p.squeeze_(0);
  lu.squeeze_(0);
  int int_info = info.squeeze_().toCInt();
  AT_CHECK(int_info >= 0, "LU factorization (getrf) failed with info = ", int_info);
  auto n = self.size(0);
  auto num_exchanges = (at::arange(1, n + 1, p.type()) != p).nonzero().size(0);
  if (num_exchanges % 2 == 1) {
    return std::make_tuple(-1., lu.diag(), int_info);
  } else {
    return std::make_tuple(1., lu.diag(), int_info);
  }
}

Tensor det(const Tensor& self) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) &&
           self.dim() == 2 && self.size(0) == self.size(1),
           "det(", self.type(), "{", self.sizes(), "}): expected a 2D square tensor "
           "of floating types");
  double det_P;
  Tensor diag_U;
  int info;
  std::tie(det_P, diag_U, info) = _lu_det_P_diag_U_info(self);
  if (info > 0) {
    return at::zeros({}, self.type());
  } else {
    return diag_U.prod().mul_(det_P);
  }
}

Tensor logdet(const Tensor& self) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) &&
           self.dim() == 2 && self.size(0) == self.size(1),
           "logdet(", self.type(), "{", self.sizes(), "}): expected a 2D square tensor "
           "of floating types");
  double det_P;
  Tensor diag_U, det;
  int info;
  std::tie(det_P, diag_U, info) = _lu_det_P_diag_U_info(self);
  if (info > 0) {
    det = at::zeros({}, self.type());
  } else {
    det = diag_U.prod().mul_(det_P);
  }
  if (det.sign().toCDouble() <= 0) {
    return det.log_();  // in order to get proper -inf (det=0) or nan (det<0)
  } else {
    return diag_U.abs().log().sum();
  }
}

std::tuple<Tensor, Tensor> slogdet(const Tensor& self) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) &&
           self.dim() == 2 && self.size(0) == self.size(1),
           "slogdet(", self.type(), "{", self.sizes(), "}): expected a 2D square tensor "
           "of floating types");
  double det_P;
  Tensor diag_U, det;
  int info;
  std::tie(det_P, diag_U, info) = _lu_det_P_diag_U_info(self);
  if (info > 0) {
    det = at::zeros({}, self.type());
  } else {
    det = diag_U.prod().mul_(det_P);
  }
  return std::make_tuple(det.sign(), diag_U.abs_().log_().sum());
}

Tensor inverse(const Tensor& self) {
  Tensor result = self.type().tensor();
  return at::native::inverse_out(result, self);
}

Tensor& inverse_out(Tensor &result, const Tensor &self) {
  AT_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "tensor should have CPU or CUDA backend");
  AT_CHECK(self.dim() == 2, "tensor should be 2 dimensional");
  AT_CHECK(self.size(0) == self.size(1), "tensor should be square");
  AT_CHECK(at::isFloatingType(self.type().scalarType()), "tensor should be of floating-point type");
  if (self.size(0) == 0) {
    return result.resize_({0, 0});
  } else {
    return at::_getri_out(result, self);
  }
}

Tensor pinverse(const Tensor& self, double rcond) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) && self.dim() == 2,
           "pinverse(", self.type(), "{", self.sizes(), "}): expected a 2D tensor "
           "of floating types");
  if (self.numel() == 0) {
    // Match NumPy
    return self.type().tensor({self.size(1), self.size(0)});
  }
  Tensor U, S, V;
  std::tie(U, S, V) = self.svd();
  Tensor max_val = S[0];
  Tensor S_pseudoinv = at::where(S > rcond * max_val, S.reciprocal(), at::zeros({}, self.options()));
  return V.mm(S_pseudoinv.diag().mm(U.t()));
}

static inline Tensor _matrix_rank_helper(const Tensor& self, bool symmetric) {
  Tensor S;
  if (!symmetric) {
    Tensor U, V;
    std::tie(U, S, V) = self.svd();
  } else {
    Tensor eigvecs;
    std::tie(S, eigvecs) = self.symeig();
    S = S.abs();
  }
  return S;
}

Tensor matrix_rank(const Tensor& self, double tol, bool symmetric) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) && self.dim() == 2,
           "matrix_rank(", self.type(), "{", self.sizes(), "}): expected a 2D tensor "
           "of floating types");

  Tensor S = _matrix_rank_helper(self, symmetric);
  return (S > tol).sum();
}

Tensor matrix_rank(const Tensor& self, bool symmetric) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) && self.dim() == 2,
           "matrix_rank(", self.type(), "{", self.sizes(), "}): expected a 2D tensor "
           "of floating types");

  Tensor S = _matrix_rank_helper(self, symmetric);
  double tol = _get_epsilon(self.type().scalarType()) * std::max(self.size(0), self.size(1));
  return (S > S.max().mul_(tol)).sum();
}

static void check_1d(const Tensor& t, const char* arg, const char* fn) {
 AT_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
}

Tensor ger(const Tensor& self, const Tensor& vec2) {
  check_1d(self, "self", "ger");
  check_1d(vec2, "vec2", "ger");
  return at::_ger(self, vec2);
}

Tensor& ger_out(Tensor& result, const Tensor& self, const Tensor& vec2) {
  check_1d(self, "self", "ger");
  check_1d(vec2, "vec2", "ger");
  return at::_ger_out(result, self, vec2);
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
  if (self.is_sparse()) {
    return mat2.type().addmm(at::zeros({}, mat2.type()), self, mat2, 0, 1);
  }
  return self.type()._mm(self, mat2);
}

Tensor& mm_out(Tensor& result, const Tensor& self, const Tensor& mat2) {
  if (self.is_sparse()) {
    return mat2.type().addmm_out(result, at::zeros({}, mat2.type()), self, mat2, 0, 1);
  }
  return self.type()._mm_out(result, self, mat2);
}

Tensor mv(const Tensor& self, const Tensor& vec) {
  check_1d(vec, "vec", "mv");
  return at::_mv(self, vec);
}

Tensor& mv_out(Tensor& result, const Tensor& self, const Tensor& vec) {
  check_1d(vec, "vec", "mv");
  return at::_mv_out(result, self, vec);
}

Tensor addmv(const Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
  check_1d(vec, "vec", "addmv");
  return at::_addmv(self, mat, vec, beta, alpha);
}

Tensor& addmv_(Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
  check_1d(vec, "vec", "addmv");
  return self._addmv_(mat, vec, beta, alpha);
}

Tensor& addmv_out(Tensor &result, const Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
  check_1d(vec, "vec", "addmv");
  return at::_addmv_out(result, self, mat, vec, beta, alpha);
}

Tensor addr(const Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");
  return at::_addr(self, vec1, vec2, beta, alpha);
}

Tensor& addr_(Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");
  return self._addr_(vec1, vec2, beta, alpha);
}

Tensor& addr_out(Tensor &result, const Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");
  return at::_addr_out(result, self, vec1, vec2, beta, alpha);
}

Tensor dot(const Tensor& self, const Tensor& tensor) {
  check_1d(self, "self", "dot");
  check_1d(tensor, "tensor", "dot");
  return self._dot(tensor);
}

Tensor& dot_out(Tensor& result, const Tensor& self, const Tensor& tensor) {
  result.resize_({});
  // dispatching through type ensures we don't allow mismatched types.
  return self.type().fill_(result, self.dot(tensor));
}

/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are broadcasted (and thus
  must be broadcastable).  For example, if tensor1 is a (j x 1 x n x m) Tensor
  and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.
*/
Tensor matmul(at::optional<Tensor> out_opt, const Tensor& tensor1, const Tensor& tensor2) {
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();
  auto has_out = out_opt.has_value();
  Tensor out = out_opt.value_or(Tensor());

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return has_out ? at::native::dot_out(out, tensor1, tensor2) : tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return has_out ? at::native::mv_out(out, tensor1, tensor2) : tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return has_out ? at::native::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
                   : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return has_out ? at::native::mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // optimization: use mm instead of bmm by folding tensor1's batch into
    // its leading matrix dimension.

    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
    auto size1 = tensor1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    if (dim_tensor2 > 1) {
      output_size.push_back(size2[dim_tensor2 - 1]);
    }

    // fold the batch into the first dimension
    Tensor t1 = tensor1.contiguous().view({-1, size1[size1.size() - 1]});
    Tensor output = has_out ? at::_unsafe_view(at::mm_out(out, t1, t2), output_size)
                            : at::_unsafe_view(t1.mm(t2), output_size);
    return has_out ? out.set_(output) : output;
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error messages
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    IntList batch_tensor1(tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
    int64_t p = tensor2.size(-1);
    IntList batch_tensor2(tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

    int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(),
                                               1, std::multiplies<int64_t>());

    std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

    std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
    tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});

    // flatten expanded batches
    Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    // reshape batches back into result
    std::vector<int64_t> output_shape(expand_batch_portion);
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    Tensor output = has_out ? at::_unsafe_view(at::bmm_out(out, tensor1_expanded, tensor2_expanded), output_shape)
                            : at::_unsafe_view(tensor1_expanded.bmm(tensor2_expanded), output_shape);

    return has_out ? out.set_(output) : output;
  }

 AT_ERROR("both arguments to matmul need to be at least 1D, but they are ",
          dim_tensor1, "D and ", dim_tensor2, "D");

}

Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  return at::native::matmul(at::nullopt, tensor1, tensor2);
}

Tensor& matmul_out(Tensor &result, const Tensor & tensor1, const Tensor & tensor2) {
  at::native::matmul(at::optional<Tensor>(result), tensor1, tensor2);
  return result;
}

}
}
