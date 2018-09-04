#include "torch/csrc/jit/type.h"

#include "torch/csrc/jit/assertions.h"

#include <iostream>

namespace torch { namespace jit {

std::ostream& operator<<(std::ostream & out, const Type & t) {
  if(auto value = t.cast<CompleteTensorType>()) {
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
  } else if (auto value = t.cast<TensorType>()) {
    out << at::toString(value->scalarType()) << "(";
    for (int i = 0; i < value->dim(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << "*";
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
  } else if(t.kind() == TypeKind::NoneType) {
    out << "None";
  } else if(t.kind() == TypeKind::StringType) {
    out << "string";
  } else if(t.kind() == TypeKind::GeneratorType) {
    out << "Generator";
  } else {
    AT_ERROR("unknown type kind");
  }
  return out;
}

DynamicTypePtr DynamicType::get() {
  static auto value = DynamicType::create();
  return value;
}
NumberTypePtr NumberType::get() {
  static auto value = NumberType::create();
  return value;
}
IntTypePtr IntType::get() {
  static auto value = IntType::create();
  return value;
}
FloatTypePtr FloatType::get() {
  static auto value = FloatType::create();
  return value;
}
NoneTypePtr NoneType::get() {
  static auto value = NoneType::create();
  return value;
}
GeneratorTypePtr GeneratorType::get() {
  static auto value = GeneratorType::create();
  return value;
}
StringTypePtr StringType::get() {
  static auto value = StringType::create();
  return value;
}
ListTypePtr ListType::ofTensors() {
  static auto value = ListType::create(DynamicType::get());
  return value;
}
ListTypePtr ListType::ofInts() {
  static auto value = ListType::create(IntType::get());
  return value;
}
ListTypePtr ListType::ofFloats() {
  static auto value = ListType::create(FloatType::get());
  return value;
}

TypePtr inferTypeFrom(const IValue& value) {
  if (value.isTensor()) {
    return CompleteTensorType::create(value.toTensor());
  } else if (value.isDouble()) {
    return FloatType::get();
  } else if (value.isInt()) {
    return IntType::get();
  } else if (value.isString()) {
    return StringType::get();
  } else if (value.isIntList()) {
    return ListType::ofInts();
  } else if (value.isTensorList()) {
    return ListType::ofTensors();
  } else if (value.isDoubleList()) {
    return ListType::ofFloats();
  } else if (value.isTuple()) {
    return TupleType::create(fmap(value.toTuple()->elements(), inferTypeFrom));
  }
  AT_ASSERTM(false, "Unhandled IValue kind in inferTypeFrom");
}

at::optional<TypePtr> unifyTypes(const TypePtr& t1, const TypePtr& t2) {
  //cases that t1 == t2, or t1 is a type refinement of t2 and vice versa
  if (t1->isSubtypeOf(t2)) {
    return t2;
  } else if (t2->isSubtypeOf(t1)) {
    return t1;
  }

  // NB: we do not return NumberType because there is not currently enough
  // operator support for it

  if (t1->isSubtypeOf(DynamicType::get()) && t2->isSubtypeOf(DynamicType::get())) {
    return static_cast<TypePtr>(DynamicType::get());;
  }

  //types which contain other types
  if (t1->cast<ListType>() && t2->cast<ListType>()) {
    auto unified_type = unifyTypes(t1->cast<ListType>()->getElementType(), t2->cast<ListType>()->getElementType());
    if (unified_type) {
      return static_cast<TypePtr>(ListType::create(*unified_type));
    } else {
      return at::nullopt;
    }
  } else if(t1->cast<TupleType>() && t2->cast<TupleType>()) {
    auto tuple1 = t1->cast<TupleType>();
    auto tuple2 = t2->cast<TupleType>();
    if (tuple1->elements().size() != tuple2->elements().size()) {
      return at::nullopt;
    }
    std::vector<TypePtr> elements;
    for (size_t i = 0; i < tuple1->elements().size(); i++) {
      if (auto elem = unifyTypes(tuple1->elements().at(i), tuple2->elements().at(i))) {
        elements.push_back(*elem);
      } else {
        return at::nullopt;
      }
    }
    return static_cast<TypePtr>(TupleType::create(elements));
  }

  return at::nullopt;
}

}} // namespace torch::jit
