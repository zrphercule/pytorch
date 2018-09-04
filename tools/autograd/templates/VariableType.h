#pragma once

// ${generated_comment}

#include <ATen/ATen.h>

#include <ATen/TypeDefault.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint> // for size_t
#include <functional> // for function
#include <memory> // for unique_ptr
#include <string>
#include <vector>

namespace torch { namespace autograd {

struct Variable;
using at::Context;
using at::Generator;
using at::IntList;
using at::Scalar;
using at::SparseTensorRef;
using at::ScalarType;
using at::Storage;
using at::Tensor;
using at::TensorList;
using at::Type;
using at::ScalarType;
using at::optional;

void register_variable_type_for(at::Type* baseType);

struct TORCH_API VariableType final : public at::TypeDefault {
  VariableType(Context* context, at::Type* baseType);
  virtual at::ScalarType scalarType() const override;
  virtual at::Backend backend() const override;
  virtual bool is_cuda() const override;
  virtual bool is_sparse() const override;
  virtual bool is_distributed() const override;
  virtual Storage storage(bool resizable = false) const override;
  virtual Storage storage(size_t size, bool resizable = false) const override;
  virtual Storage storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const override;
  virtual Storage storageWithAllocator(int64_t size, at::Allocator* allocator) const override;
  virtual std::unique_ptr<at::Generator> generator() const override;
  virtual const char * toString() const override;
  virtual at::TypeID ID() const override;
  virtual size_t elementSizeInBytes() const override;
  virtual at::Type & toBackend(at::Backend b) const override;
  virtual at::Type & toScalarType(at::ScalarType s) const override;
  static const char * typeString();
  virtual Storage unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  virtual at::Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;

  static at::Type* getVariableTypeFromBaseType(const at::Type& baseType);
  static bool isVariableType(const at::Type& type);
  static std::vector<at::Type*> allCUDATypes();
  static std::vector<at::Type*> allCPUTypes();

  virtual Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override;
  virtual Tensor & _s_copy_from(const Tensor & self, Tensor & dst, bool non_blocking) const override;
  ${type_derived_method_declarations}

private:
  // checks that t is actually a Variable
  static Variable & checked_cast_variable(const Tensor & t, const char * name, int pos);
  static at::Tensor & unpack(const Tensor & t, const char * name, int pos);
  static at::SparseTensorRef unpack(SparseTensorRef t, const char * name, int pos);
  static at::Tensor unpack_opt(const Tensor & t, const char * name, int pos);
  static std::vector<at::Tensor> unpack(at::TensorList tl, const char *name, int pos);

  at::Type* baseType;
  std::string str;
  size_t id_;
};

}} // namespace torch::autograd
