#include "ATen/ATen.h"
#include "torch/csrc/jit/script/lexer.h"
#include "torch/csrc/jit/script/tree.h"
#include "torch/csrc/jit/operator.h"

#include "torch/csrc/jit/script/error_report.h"

namespace torch { namespace jit {

namespace script {
struct SchemaParser {
  SchemaParser(const std::string& str)
  : L(str) {}

  FunctionSchema parseDeclaration() {
    auto name = L.expect(TK_IDENT).text();
    if(L.nextIf(':')) {
      L.expect(':');
      name = name + "::" + L.expect(TK_IDENT).text();
    }
    std::vector<Argument> arguments;
    std::vector<Argument> returns;
    kwarg_only = false;
    parseList('(', ',', ')', arguments, &SchemaParser::parseArgument);
    L.expect(TK_ARROW);
    if(L.cur().kind == '(') {
      parseList('(', ',', ')', returns, &SchemaParser::parseReturn);
    } else {
      parseReturn(returns);
    }
    return FunctionSchema { name, arguments, returns };
  }

  std::vector<FunctionSchema> parseDeclarations() {
    std::vector<FunctionSchema> results;
    do {
      results.push_back(parseDeclaration());
    } while(L.nextIf(TK_NEWLINE));
    L.expect(TK_EOF);
    return results;
  }

  TreeRef parseIdent() {
    return String::create(L.expect(TK_IDENT).text());
  }
  TypePtr parseBaseType() {
    static std::unordered_map<std::string, TypePtr> type_map = {
      {"Tensor", DynamicType::get() },
      {"Generator", DynamicType::get() },
      {"ScalarType", IntType::get() },
      {"Layout", IntType::get() },
      {"Device", ListType::ofInts() },
      {"Scalar", NumberType::get() },
      {"float", FloatType::get() },
      {"int", IntType::get() },
      {"bool", IntType::get() }, // TODO: add separate bool type
    };
    auto tok = L.expect(TK_IDENT);
    auto text = tok.text();
    auto it = type_map.find(text);
    if(it == type_map.end())
      throw ErrorReport(tok.range) << "unknown type specifier";
    return it->second;
  }
  void parseType(Argument& arg) {
    arg.type = parseBaseType();
    if(L.nextIf('[')) {
      arg.type = std::make_shared<ListType>(arg.type);
      if(L.cur().kind == TK_NUMBER) {
        arg.N = std::stoll(L.next().text());
      }
      L.expect(']');
    }
  }

  void parseArgument(std::vector<Argument>& arguments) {
    // varargs
    if(L.nextIf('*')) {
      kwarg_only = true;
      return;
    }
    Argument arg;
    parseType(arg);

    // nullability is ignored for now, since the JIT never cares about it
    L.nextIf('?');
    arg.name = L.expect(TK_IDENT).text();
    if(L.nextIf('=')) {
      parseDefaultValue(arg);
    }
    arg.kwarg_only = kwarg_only;
    arguments.push_back(std::move(arg));
  }
  void parseReturn(std::vector<Argument>& args) {
    Argument arg("ret" + std::to_string(args.size()));
    parseType(arg);
    args.push_back(std::move(arg));
  }
  IValue parseSingleConstant(TypeKind kind) {
    switch(L.cur().kind) {
      case TK_TRUE:
        L.next();
        return true;
      case TK_FALSE:
        L.next();
        return false;
      case TK_IDENT: {
        auto tok = L.next();
        auto text = tok.text();
        if("float" == text) {
          return static_cast<int64_t>(at::kFloat);
        } else if("cpu" == text) {
          return static_cast<int64_t>(at::Device::Type::CPU);
        } else if("strided" == text) {
          return static_cast<int64_t>(at::kStrided);
        } else if("ElementwiseMean" == text) {
          return static_cast<int64_t>(Reduction::ElementwiseMean);
        } else {
          throw ErrorReport(L.cur().range) << "invalid numeric default value";
        }
      }
      default:
        std::string n;
        if(L.nextIf('-'))
          n = "-" + L.expect(TK_NUMBER).text();
        else
          n = L.expect(TK_NUMBER).text();
        if(kind == TypeKind::FloatType || n.find(".") != std::string::npos || n.find("e") != std::string::npos) {
          return std::stod(n);
        } else {
          int64_t v = std::stoll(n);
          return v;
        }
    }
  }
  IValue convertToList(TypeKind kind, const SourceRange& range, std::vector<IValue> vs) {
    switch(kind) {
        case TypeKind::FloatType:
          return fmap(vs, [](IValue v) {
            return v.toDouble();
          });
        case TypeKind::IntType:
          return fmap(vs, [](IValue v) {
            return v.toInt();
          });
        default:
          throw ErrorReport(range) << "lists are only supported for float or int types.";
      }
  }
  IValue parseConstantList(TypeKind kind) {
    auto tok = L.expect('[');
    std::vector<IValue> vs;
    if(L.cur().kind != ']') {
      do {
        vs.push_back(parseSingleConstant(kind));
      } while(L.nextIf(','));
    }
    L.expect(']');
    return convertToList(kind, tok.range, std::move(vs));
  }

  IValue parseTensorDefault(const SourceRange& range) {
    if("None" == L.expect(TK_IDENT).text()) {
      return at::Tensor();
    } else {
      throw ErrorReport(range) << "invalid tensor default value";
    }
  }
  void parseDefaultValue(Argument& arg) {
    auto range = L.cur().range;
    switch(arg.type->kind()) {
      case TypeKind::DynamicType: {
        arg.default_value = parseTensorDefault(range);
      }  break;
      case TypeKind::NumberType:
      case TypeKind::IntType:
      case TypeKind::FloatType:
        arg.default_value = parseSingleConstant(arg.type->kind());
        break;
      case TypeKind::ListType: {
        auto elem_kind = arg.type->cast<ListType>()->getElementType();
        if(L.cur().kind == TK_IDENT) {
          arg.default_value = parseTensorDefault(range);
        } else if(arg.N && L.cur().kind != '[') {
          IValue v = parseSingleConstant(elem_kind->kind());
          std::vector<IValue> repeated(*arg.N, v);
          arg.default_value = convertToList(elem_kind->kind(), range, repeated);
        } else {
          arg.default_value = parseConstantList(elem_kind->kind());
        }
      } break;
      default:
        throw ErrorReport(range) << "unexpected type, file a bug report";
    }
  }

  template<typename T>
  void parseList(int begin, int sep, int end, std::vector<T>& result, void (SchemaParser::*parse)(std::vector<T>&)) {
    auto r = L.cur().range;
    if (begin != TK_NOTHING)
      L.expect(begin);
    if (L.cur().kind != end) {
      do {
        (this->*parse)(result);
      } while (L.nextIf(sep));
    }
    if (end != TK_NOTHING)
      L.expect(end);
  }
  Lexer L;
  bool kwarg_only;
};
}


namespace {

std::string canonicalSchemaString(const FunctionSchema& schema) {
  std::ostringstream out;

  out << schema.name;
  out << "(";

  bool seen_kwarg_only = false;
  for(size_t i = 0; i < schema.arguments.size(); ++i) {
    if (i > 0) out << ", ";
    if (schema.arguments[i].kwarg_only && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    const auto & arg = schema.arguments[i];
    out << arg.type->str() << " " << arg.name;
  }

  out << ") -> ";
  if (schema.returns.size() == 1) {
    out << schema.returns.at(0).type->str();
  } else if (schema.returns.size() > 1) {
    out << "(";
    for (size_t i = 0; i < schema.returns.size(); ++i) {
      if (i > 0) out << ", ";
      out << schema.returns[i].type->str();
    }
    out << ")";
  }
  return out.str();
}

using OperatorMap = std::unordered_map<Symbol, std::vector<std::shared_ptr<Operator>>>;
struct OperatorRegistry  {
  OperatorMap operators;
  std::mutex lock;
  // Those two maps are used to implement lookupByLiteral, which is needed for the n->match(...) calls.
  // Basically, every function schema is assigned a unique string you can use to match it. However,
  // parsing those strings or comparing and hashing them character by character would be very slow, so
  // we use a trick here! Every string literal in your program is guaranteed to have static storage
  // duration and so its address won't change at runtime. This allows us to memoize answerts for every
  // pointer, which is done by the operators_by_sig_literal map. Still, this map is initially
  // empty, and so we still need to do the complete string matching at the first time, which is implemented
  // by performing a lookup in the operators_by_sig map.
  std::unordered_map<std::string, std::shared_ptr<Operator>> operators_by_sig;
  std::unordered_map<const char *, std::shared_ptr<Operator>> operators_by_sig_literal;
  void registerOperator(Operator&& op){
    std::lock_guard<std::mutex> guard(lock);

    Symbol sym = Symbol::fromQualString(op.schema.name);
    auto op_ptr = std::make_shared<Operator>(std::move(op));

    operators[sym].push_back(op_ptr);

    operators_by_sig[canonicalSchemaString(op.schema)] = op_ptr;
  }

  Operator& lookupByLiteral(const char * name) {
    auto it = operators_by_sig_literal.find(name);
    if (it == operators_by_sig_literal.end()) {
      auto op_ptr_it = operators_by_sig.find(name);
      // Handy debugging code that dumps all operators we know about on mismatch
#if 0
      if (op_ptr_it == operators_by_sig.end()) {
        for (auto & entry : operators_by_sig) {
          std::cout << entry.first << std::endl;
        }
      }
#endif
      JIT_ASSERTM(op_ptr_it != operators_by_sig.end(), "Couldn't find an operator for %s", name);
      it = operators_by_sig_literal.emplace_hint(it, name, op_ptr_it->second);
    }
    return *it->second;
  }

  const std::vector<std::shared_ptr<Operator>>& getOperators(Symbol name) {
    std::lock_guard<std::mutex> guard(lock);
    static std::vector<std::shared_ptr<Operator>> empty;
    auto it = operators.find(name);
    if(it != operators.end())
      return it->second;
    return empty;
  }
};

OperatorRegistry& getRegistry() {
  static OperatorRegistry r;
  return r;
}

} // anonymous namespace

void registerOperator(Operator&& op) {
  getRegistry().registerOperator(std::move(op));
}

const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(Symbol name) {
  return getRegistry().getOperators(name);
}

Operator& sig(const char *signature) {
  return getRegistry().lookupByLiteral(signature);
}

FunctionSchema parseSchema(const std::string& schema) {
  return script::SchemaParser(schema).parseDeclarations().at(0);
}

at::optional<AttributeKind> attributeKindOf(TypePtr type) {
  switch(type->kind()) {
    case TypeKind::IntType: return AttributeKind::i;
    case TypeKind::FloatType: return AttributeKind::f;
    case TypeKind::NumberType: return AttributeKind::t;
    case TypeKind::ListType:
      if(type->isSubtypeOf(*ListType::ofInts()))
        return AttributeKind::is;
      else
        return at::nullopt;
    default:
      return at::nullopt;
  }
}

bool typeMatches(TypePtr actual, TypePtr formal) {
  return actual->isSubtypeOf(*formal);
}

bool Operator::matches(const Node* node) const {
  if (node->kind().toQualString() != schema.name) {
    return false;
  }
  size_t attributes_size = node->numAttributes();
  size_t attributes_seen = 0;
  auto inputs_size = node->inputs().size();
  size_t input_i = 0;
  for(size_t arg_i = 0; arg_i < schema.arguments.size(); ++arg_i) {
    at::optional<AttributeKind> attribute_kind;
    const Argument& arg = schema.arguments[arg_i];
    if(attributes_size > 0 && (attribute_kind = attributeKindOf(arg.type))) {
      auto name = Symbol::fromQualString("attr::" + arg.name);
      if(!node->hasAttribute(name) || node->kindOf(name) != *attribute_kind) {
        // std::cout << "missing attribute: " << name << "\n";
        return false;
      }
      attributes_seen++;
    } else if(*arg.type == *ListType::ofTensors()) {
      // Tensor[] is handled as varargs, consume inputs until the remaining required arguments
      // XXX - there can only be a single Tensor[] in a declaration
      size_t remaining_required = 0;
      for(size_t j = arg_i + 1; j < schema.arguments.size(); ++j){
        // remaining arguments are only those that won't be consumed from attributes
        if(attributes_size == 0 || !attributeKindOf(schema.arguments[j].type))
          remaining_required++;
      }
      while(inputs_size - input_i > remaining_required) {
        auto input = node->inputs()[input_i++];
        if(!typeMatches(input->type(), DynamicType::get())) {
          // std::cout << "vararg argument is not Dynamic\n";
          return false;
        }
      }
    } else {
      if(input_i == inputs_size) {
        // std::cout << "not enough inputs\n";
        return false;
      }
      auto input = node->inputs()[input_i++];
      if(!typeMatches(input->type(), arg.type)) {
        // std::cout << "argument " << arg_i << " has the wrong type\n";
        return false;
      }
    }
  }

  if(!schema.is_vararg && input_i != inputs_size) {
    // std::cout << "not all inputs used\n" << input_i << " " << inputs_size << "\n";
    return false;
  }
  if(!schema.is_vararg && attributes_seen != attributes_size) {
    // std::cout << "not all attributes used\n" << attributes_seen << " " << attributes_size << "\n";
    return false;
  }
  return true;
}

std::shared_ptr<Operator> findOperatorFor(const Node* node) {
  const auto& candidates = getAllOperatorsFor(node->kind());
  for(const auto& candidate : candidates) {
    if(candidate->matches(node)) {
      return candidate;
    }
  }
  return nullptr;
}

const Operator& getOperatorFor(const Node* node) {
  auto op = findOperatorFor(node);
  if(op)
    return *op;

  auto er = script::ErrorReport(node->getSourceLocation());
  er << "Schema not found for node. File a bug report.\n";
  er << "Node: " << *node << "\n";
  er << "Input types:";
  for(size_t i = 0; i < node->inputs().size(); ++i) {
    if(i > 0)
      er << ", ";
    er << *node->inputs()[i]->type();
  }
  er << "\ncandidates were:\n";
  const auto& candidates = getAllOperatorsFor(node->kind());
  for(auto & candidate : candidates) {
    er << "  " << candidate->schema << "\n";
  }
  throw er;
}

}}
