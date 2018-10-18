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
    bool kwarg_only = false;
    size_t idx = 0;
    parseList('(', ',', ')', [&] {
      if(L.nextIf('*')) {
        kwarg_only = true;
      } else {
        arguments.push_back(parseArgument(
            idx++, /*is_return=*/false, /*kwarg_only=*/kwarg_only));
      }
    });
    idx = 0;
    L.expect(TK_ARROW);
    if (L.cur().kind == '(') {
      parseList('(', ',', ')', [&] {
        returns.push_back(
            parseArgument(idx++, /*is_return=*/true, /*kwarg_only=*/false));
      });
    } else {
      returns.push_back(
          parseArgument(0, /*is_return=*/true, /*kwarg_only=*/false));
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
      {"Generator", GeneratorType::get() },
      {"ScalarType", IntType::get() },
      {"Layout", IntType::get() },
      {"Device", ListType::ofInts() },
      {"Scalar", NumberType::get() },
      {"str", StringType::get() },
      {"float", FloatType::get() },
      {"int", IntType::get() },
      {"bool", BoolType::get() },
      {"World", WorldType::get() },
    };
    auto tok = L.expect(TK_IDENT);
    auto text = tok.text();
    auto it = type_map.find(text);
    if(it == type_map.end()) {
      if(text.size() > 0 && islower(text[0])) {
        // lower case identifiers that are not otherwise valid types
        // are treated as type variables
        return VarType::create(text);
      }
      throw ErrorReport(tok.range) << "unknown type specifier";
    }
    return it->second;
  }
  TypePtr parseType() {
    TypePtr value;
    if (L.cur().kind == '(') {
      std::vector<TypePtr> types;
      parseList('(', ',', ')', [&]{
        types.push_back(parseType());
      });
      value = TupleType::create(std::move(types));
    } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Future") {
      L.next(); // Future
      L.expect('(');
      TypePtr subtype = parseType();
      L.expect(')');
      throw ErrorReport(L.cur()) << "Futures are not yet implemented";
    } else if (L.cur().kind == TK_IDENT && L.cur().text() == "Tensor") {
      value = DynamicType::get();
      auto range = L.next().range; // Tensor
      if(L.nextIf('(')) {
        // optional 'alias set annotation'
        auto bt = parseBaseType();
        if (bt->kind() != TypeKind::VarType) {
          throw ErrorReport(range)
              << "expected type variable but found " << bt->str();
        }
        L.expect(')');
        // annotation itself is currently unused
      }
    } else {
      value = parseBaseType();
    }
    while(true) {
      if(L.cur().kind == '[' && L.lookahead().kind == ']') {
        L.next(); // [
        L.next(); // ]
        value = ListType::create(value);
      } else if(L.nextIf('?')) {
        // pass - currently ignored but optional types would go here when implemented
      } else {
        break;
      }
    }
    return value;
  }

  Argument parseArgument(size_t idx, bool is_return, bool kwarg_only) {
    Argument result;
    result.type = parseType();
    if(L.nextIf('[')) {
      // note: an array with a size hint can only occur at the Argument level
      result.type = ListType::create(result.type);
      result.N = std::stoll(L.expect(TK_NUMBER).text());
      L.expect(']');
    }
    if(L.nextIf('!')) {
      // pass - currently ignored but will be used to annotate where we write to a
      // tensor
    }
    if(is_return) {
      // optionally named return values
      if(L.cur().kind == TK_IDENT) {
        result.name = L.next().text();
      } else {
        result.name = "ret" + std::to_string(idx);
      }
    } else {
      result.kwarg_only = kwarg_only;
      result.name = L.expect(TK_IDENT).text();
      if(L.nextIf('=')) {
        parseDefaultValue(result);
      }
    }
    return result;
  }
  IValue parseSingleConstant(TypeKind kind) {
    switch(L.cur().kind) {
      case TK_TRUE:
        L.next();
        return true;
      case TK_FALSE:
        L.next();
        return false;
      case TK_NONE:
        L.next();
        return IValue();
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
        case TypeKind::BoolType:
          return fmap(vs, [](IValue v) {
            return v.toBool();
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
    L.expect(TK_NONE);
    return IValue();
  }
  void parseDefaultValue(Argument& arg) {
    auto range = L.cur().range;
    switch(arg.type->kind()) {
      case TypeKind::DynamicType:
      case TypeKind::GeneratorType: {
        arg.default_value = parseTensorDefault(range);
      }  break;
      case TypeKind::NumberType:
      case TypeKind::IntType:
      case TypeKind::BoolType:
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

  void parseList(int begin, int sep, int end, std::function<void()> callback) {
    auto r = L.cur().range;
    if (begin != TK_NOTHING)
      L.expect(begin);
    if (L.cur().kind != end) {
      do {
        callback();
      } while (L.nextIf(sep));
    }
    if (end != TK_NOTHING)
      L.expect(end);
  }
  Lexer L;
};

} // namespace script

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
private:
  std::mutex lock;
  OperatorMap operators;
  // list of operators whose schema have not yet been parsed, and must
  // be registered before any call to lookup an opeator
  std::vector<std::shared_ptr<Operator>> to_register;
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

  // XXX - caller must be holding lock
  void registerPendingOperators() {
    for(auto op : to_register) {
      Symbol sym = Symbol::fromQualString(op->schema().name);
      operators[sym].push_back(op);
      operators_by_sig[canonicalSchemaString(op->schema())] = op;
    }
    to_register.clear();
  }

public:
  void registerOperator(Operator&& op) {
    std::lock_guard<std::mutex> guard(lock);
    to_register.push_back(std::make_shared<Operator>(std::move(op)));
  }

  const std::shared_ptr<Operator>& lookupByLiteral(const char * name) {
    std::lock_guard<std::mutex> guard(lock);
    registerPendingOperators();
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
      JIT_ASSERTM(op_ptr_it != operators_by_sig.end(), "Couldn't find an operator for ", name);
      it = operators_by_sig_literal.emplace_hint(it, name, op_ptr_it->second);
    }
    return it->second;
  }


  const std::vector<std::shared_ptr<Operator>>& getOperators(Symbol name) {
    std::lock_guard<std::mutex> guard(lock);
    registerPendingOperators();
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
  return *getRegistry().lookupByLiteral(signature);
}

FunctionSchema parseSchema(const std::string& schema) {
  return script::SchemaParser(schema).parseDeclarations().at(0);
}

bool Operator::matches(const Node* node) const {
  // wrong name
  if (node->kind().toQualString() != schema().name) {
    return false;
  }
  at::ArrayRef<const Value*> actuals = node->inputs();
  const auto& formals = schema().arguments;

  // not enough inputs
  if(actuals.size() < formals.size())
    return false;


  TypeEnv type_env;
  for(size_t i = 0; i < formals.size(); ++i) {
    try {
      TypePtr formal = matchTypeVariables(formals[i].type, actuals[i]->type(), type_env);
      // mismatched input type
      if (!actuals[i]->type()->isSubtypeOf(formal)) {
        return false;
      }
    } catch(TypeMatchError& err) {
      return false;
    }
  }

  // too many inputs
  if(!schema().is_vararg && actuals.size() != formals.size()) {
    // std::cout << "not all inputs used\n" << input_i << " " << inputs_size << "\n";
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
    er << "  " << candidate->schema() << "\n";
  }
  throw er;
}


OperatorSet::OperatorSet(std::initializer_list<const char *> sig_literals) {
  auto & registry = getRegistry();
  for (const char * sig : sig_literals) {
    auto op = registry.lookupByLiteral(sig);
    ops[Symbol::fromQualString(op->schema().name)].push_back(op);
  }
}

Operator* OperatorSet::find(const Node *n) const  {
  auto it = ops.find(n->kind());
  if (it == ops.end()) {
    return nullptr;
  }
  for (auto & op : it->second) {
    if (op->matches(n)) {
      return op.get();
    }
  }
  return nullptr;
}

}}
