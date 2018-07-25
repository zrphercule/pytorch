#include "torch/csrc/jit/type.h"

#include "torch/csrc/jit/assertions.h"

#include <iostream>

namespace torch { namespace jit {

std::ostream& operator<<(std::ostream & out, const Type & t) {
  if(auto value = t.cast<TensorType>()) {
    out << at::toString(value->scalarType()) << "(";
    auto& sizes = value->sizes();
    auto& strides = value->strides();
    JIT_ASSERT(sizes.size() == strides.size());
    for (size_t i = 0; i < sizes.size(); i++) {
      if (i > 0) {
        out << ", ";
      }
      // TODO: figure out a good way to output strides, or
      // add a "debug" printing mode which adds the extra stuff
      out << sizes[i]; // << "%" << strides[i];
      int64_t expected = i + 1 < sizes.size() ? sizes[i+1]*strides[i+1] : 1;
      if (strides[i] != expected) {
        out << "!"; //mark non-contiguous
      }
    }
    out << ")";
  } else if(t.kind() == TypeKind::DynamicType) {
    out << "Dynamic";
  } else if(t.kind() == TypeKind::TupleType) {
    out << "Tuple";
  } else if(t.kind() == TypeKind::NumberType) {
    out << "Number";
  } else if(t.kind() == TypeKind::FloatType) {
    out << "float";
  } else if(t.kind() == TypeKind::IntType) {
    out << "int";
  } else if(t.kind() == TypeKind::ListType) {
    auto prim = t.cast<ListType>()->getElementType();
    out << *prim << "[]";
  } else {
    AT_ERROR("unknown type kind");
  }
  return out;
}

TypePtr DynamicType::get() {
  static auto value = std::make_shared<DynamicType>();
  return value;
}
TypePtr NumberType::get() {
  static auto value = std::make_shared<NumberType>();
  return value;
}
TypePtr IntType::get() {
  static auto value = std::make_shared<IntType>();
  return value;
}
TypePtr FloatType::get() {
  static auto value = std::make_shared<FloatType>();
  return value;
}


TypePtr ListType::ofTensors() {
  static auto value = std::make_shared<ListType>(DynamicType::get());
  return value;
}
TypePtr ListType::ofInts() {
  static auto value = std::make_shared<ListType>(IntType::get());
  return value;
}

}} // namespace torch::jit
