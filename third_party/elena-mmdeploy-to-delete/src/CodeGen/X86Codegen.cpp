//===-- elena/src/codegen/X86Codegen.cpp
// - Code generate for x86 code -------*- C++ -*-===//
//
// Part of the Elena Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration and implementation of the X86Codegen
/// and the TopologySorter class, which is used to generate x86 code.
///
//===----------------------------------------------------------------------===//

#include "CodeGen/DeviceCodegen.h"
#include "CodeGen/TextGen.h"
#include "IR/VisitorBase.h"
#include "api.h"

using namespace ir;  // NOLINT

class X86Codegen;

///
/// \brief Sort the topology of tensor var
class TopologySorter : public VisitorBase<TopologySorter> {
 public:
  explicit TopologySorter(X86Codegen *codegen) : codegen{codegen} {}

  void visit(TensorVar *);

  ///
  /// \brief Mark the current tensor var as be visited
  /// \param p
  void markVisited(TensorVar *p) { markVisitedByName(p->get_name()); }

  ///
  /// \brief Mark the current tensor var as be visited by name
  /// \param p
  void markVisitedByName(std::string n) { well_defined.insert(std::move(n)); }

  using VisitorBase::visit;

 private:
  std::set<std::string> well_defined;
  X86Codegen *codegen;
};

///
/// \brief Generate x86 code
class X86Codegen final : public VisitorBase<X86Codegen>, public TextGen {
 public:
  explicit X86Codegen(std::ostream &ostr) : TextGen(ostr), sorter(this) {}

  void visit(Allocate *);
  void visit(Provide *);
  void visit(For *);
  void visit(Store *);
  void visit(Attr *);
  void visit(TensorVar *);
  void visit(Binary *);
  void visit(Unary *);
  void visit(IterVar *);
  void visit(ScalarVar *);
  void visit(Let *);
  void visit(IfThenElse *);
  void visit(Logical *);
  void visit(Select *);
  void visit(Call *);
  void visit(Evaluate *);

  template <typename T>
  void visit(Const<T> *);

  // for raw code gen
  void visit(ComputeOp *);
  void visit(Reduce *);

  using VisitorBase::visit;

  TopologySorter sorter;
};

void X86Codegen::visit(Allocate *allocate_ptr) {
  CHECK_NODE_TYPE(allocate_ptr->var, TensorVar)
  auto tensor_ptr = ptr_cast<TensorVar>(allocate_ptr->var);
  sorter.markVisited(tensor_ptr.get());
  *this << TYPE_OF(tensor_ptr) << " ";
  visit(tensor_ptr);

  for (const auto &rg : allocate_ptr->bound->element) {
    *this << "[";
    visit(rg->extent);
    *this << "]";
  }

  *this << ";" << endl;

  visit(allocate_ptr->body);
}

void X86Codegen::visit(Provide *provide_ptr) {
  visit(provide_ptr->var);
  visit(provide_ptr->index);
  *this << " = ";
  visit(provide_ptr->value);
}

void X86Codegen::visit(For *for_stmt_ptr) {
  if (for_stmt_ptr->it->iter_type == ir::IterAttrType::Unrolled) {
    *this << "#pragma unroll" << endl;
  }
  *this << "for (" << TYPE_OF(for_stmt_ptr->it) << " ";
  visit(for_stmt_ptr->it);
  *this << " = ";
  visit(for_stmt_ptr->init);
  *this << "; ";
  visit(for_stmt_ptr->it);
  *this << " < ";
  visit(for_stmt_ptr->init);
  *this << " + ";
  visit(for_stmt_ptr->extent);
  *this << "; ++";
  visit(for_stmt_ptr->it);
  *this << ") " << block_begin;
  visit(for_stmt_ptr->body);
  *this << block_end;
}

void X86Codegen::visit(Store *store_ptr) {
  visit(store_ptr->var);
  for (const auto &index : store_ptr->index->element) {
    *this << "[";
    visit(index);
    *this << "]";
  }
  *this << " = ";
  visit(store_ptr->value);
  *this << ";" << endl;
}

void X86Codegen::visit(TensorVar *tensor_ptr) {
  *this << makeIdentifier(tensor_ptr->get_name());
}

void X86Codegen::visit(IterVar *iter_ptr) {
  // DISCUSS: Unlike in CUDA, all variable names are fed into
  // makeIdentifier, should some special variables be preserved?
  *this << makeIdentifier(iter_ptr->get_name());
}

void X86Codegen::visit(ScalarVar *scalar_ptr) {
  if (!scalar_ptr->is_placeholder()) {
    visit(scalar_ptr->tensor);
    for (const auto &index : scalar_ptr->indices->element) {
      *this << "[";
      visit(index);
      *this << "]";
    }
  } else {
    *this << scalar_ptr->get_name();
  }
}

void X86Codegen::visit(Attr *attr_ptr) {
  // expand a threadIdx/blockIdx into a loop
  if (attr_ptr->key == AttrType::ThreadExtent) {
    const auto iter = ptr_cast<Expr>(attr_ptr->node);
    *this << "for (" << TYPE_OF(iter) << " ";
    visit(iter);
    *this << " = 0; ";
    visit(iter);
    *this << " < ";
    visit(attr_ptr->value);
    *this << "; ++";
    visit(iter);
    *this << ")" << block_begin;
  }
  visit(attr_ptr->body);
  if (attr_ptr->key == AttrType::ThreadExtent) {
    *this << block_end;
  }
}

void X86Codegen::visit(Logical *logical_ptr) {
  *this << "(";
  visit(logical_ptr->lhs);
  *this << " " << LOGICALTYPE_SYMBOL(logical_ptr->operation_type) << " ";
  visit(logical_ptr->rhs);
  *this << ")";
}

void X86Codegen::visit(Unary *unary_ptr) {
  const auto type = unary_ptr->get_dtype();
  if (type == ir::ScalarType::Float32 &&
      unary_ptr->operation_type == UnaryType::Abs) {
    *this << "(fabs(";
  } else {
    *this << "(" << UOP_DEVICE_NAME(unary_ptr->operation_type) << "(";
  }
  visit(unary_ptr->operand);
  *this << "))";
}

void X86Codegen::visit(Binary *binary_ptr) {
  if (binary_ptr->operation_type == BinaryType::Max) {
    *this << "max(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::Min) {
    *this << "min(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else {
    *this << "(";
    visit(binary_ptr->lhs);
    *this << " " << BINARYTYPE_SYMBOL(binary_ptr->operation_type) << " ";
    visit(binary_ptr->rhs);
    *this << ")";
  }
}

template <typename T>
void X86Codegen::visit(Const<T> *const_ptr) {
  *this << std::boolalpha << const_ptr->get_value();
}

void X86Codegen::visit(Let *let_ptr) {
  // DISCUSS: see cuda_codegen::visit_Let for detail
  // *this << block_begin;
  *this << "const " << TYPE_OF(let_ptr->var) << " ";
  visit(let_ptr->var);
  *this << " = ";
  visit(let_ptr->value);
  *this << ";" << endl;
  visit(let_ptr->body);
  // *this << block_end;
}

void X86Codegen::visit(IfThenElse *if_then_else_ptr) {
  *this << "if (";
  visit(if_then_else_ptr->condition);
  *this << ") " << block_begin;
  visit(if_then_else_ptr->then_case);
  *this << block_end;
  if (if_then_else_ptr->else_case) {
    *this << " else " << block_begin;
    visit(if_then_else_ptr->else_case);
    *this << block_end;
  }
}

void X86Codegen::visit(Select *select_ptr) {
  *this << "(";
  visit(select_ptr->cond);
  *this << " ? ";
  visit(select_ptr->tBranch);
  *this << " : ";
  visit(select_ptr->fBranch);
  *this << ")";
}

void X86Codegen::visit(Call *call_ptr) {
  if (call_ptr->func == CallFunction::Sync) {
    // DISCUSS: Seems that no 'sync' is needed
    *this << "/* sync() */";
  } else {
    throw std::runtime_error(
        "Calling function other than 'Sync' and 'Select' is not supported!");
  }
}

void X86Codegen::visit(Evaluate *evaluate_ptr) {
  visit(evaluate_ptr->value);
  *this << ";" << endl;
}

void X86Codegen::visit(ComputeOp *compute) {
  for (auto &i : compute->iter_vars->element) {
    *this << "for (" << TYPE_OF(i) << " ";
    visit(i);
    *this << " = ";
    visit(i->range->init);
    *this << "; ";
    visit(i);
    *this << " < ";
    visit(i->range->extent);
    *this << "; ++";
    visit(i);
    *this << ") " << block_begin;
  }
  auto print_assignment_head = [&] {
    visit(compute->output(0));
    for (auto &i : compute->iter_vars->element) {
      *this << '[';
      visit(i);
      *this << ']';
    }
    *this << " = ";
  };
  if (compute->fcompute->get_type() == IRNodeType::Reduce) {
    auto reduce = static_cast<Reduce *>(compute->fcompute.get());
    visit(reduce);
    print_assignment_head();
    visit(reduce->accumulate);
    *this << ";" << endl;
  } else {
    print_assignment_head();
    visit(compute->fcompute);
    *this << ";" << endl;
  }
  for (auto &i : compute->iter_vars->element) {
    *this << block_end;
    // 'i' is deliberately not used.
    static_cast<void>(i);
  }
}

void X86Codegen::visit(Reduce *reduce) {
  *this << TYPE_OF(reduce->accumulate) << " ";
  visit(reduce->accumulate);
  *this << " = ";
  visit(reduce->init);
  *this << ";" << endl;
  for (auto &i : reduce->reduce_axis->element) {
    *this << "for (" << TYPE_OF(i) << " ";
    visit(i);
    *this << " = ";
    visit(i->range->init);
    *this << "; ";
    visit(i);
    *this << " < ";
    visit(i->range->extent);
    *this << "; ++";
    visit(i);
    *this << ") " << block_begin;
  }
  visit(reduce->accumulate);
  *this << " = ";
  visit(reduce->combiner);
  *this << ";" << endl;
  for (auto &i : reduce->reduce_axis->element) {
    *this << block_end;
    // 'i' is deliberately not used.
    static_cast<void>(i);
  }
}

// void TopologySorter::visit(TensorVar *node) {
//  if (WellDefined.find(node->get_name()) != WellDefined.cend()) return;
//  markVisited(node);
//  // recursively visit children
//  visit(node->op);
//  // define the tensor and initialize to zeros.
//  *CodeGenerator << TYPE_OF(node) << " ";
//  CodeGenerator->visit(node);
//  for (const auto &n : node->shape->element) {
//    *CodeGenerator << '[';
//    CodeGenerator->visit(n);
//    *CodeGenerator << ']';
//  }
//  *CodeGenerator << " = {0};" << endl;
//  // expand the op to 'for' loops.
//  CodeGenerator->visit(node->op);
//}
//
// void TopologySorter::markVisited(TensorVar *node) {
//  markVisitedByName(node->get_name());
//}
//
// void TopologySorter::markVisitedByName(std::string n) {
//  WellDefined.insert(std::move(n));
//}

// void api::genX86Src(const NodePtr &node,
//                    const std::vector<std::pair<int, ScalarType>> &arg_list,
//                    std::string kernel_name, std::ostream &ostr) {
//  X86Codegen visitor{ostr};
//  for (auto &arg : arg_list)
//    visitor.sorter.markVisitedByName("Var" + std::to_string(arg.first));
//
//  visitor << Prelude;
//  visitor << "void " << kernel_name << "(";
//
//  for (int i = 0; i < arg_list.size() - 1; ++i) {
//    std::string str = SCALARTYPE_SYMBOL(arg_list[i].second);
//    if (str == "bool") str = "float";
//    visitor << str << "* __restrict__ Var" << arg_list[i].first << ", ";
//  }
//
//  std::string str = SCALARTYPE_SYMBOL(arg_list[arg_list.size() - 1].second);
//  if (str == "bool" && arg_list.size() == 1) str = "float";
//  visitor << str << "* __restrict__ Var" << arg_list[arg_list.size() -
//  1].first;
//  visitor << ") " << block_begin;
//
//  if (node == nullptr) {
//    visitor << block_end;
//    return;
//  }
//  auto cur = node;
//  while (cur->get_type() == IRNodeType::Attr)
//    cur = ptr_cast<Attr, Node>(cur)->body;
//  ELENA_ASSERT_EQ(cur->get_type(), IRNodeType::Allocate, "Node type
//  mismatch.")
//
//  auto anode = ptr_cast<Allocate, Node>(cur);
//
//  visitor.visit(anode->body);
//  visitor << block_end;
//}
//
// std::string api::genX86Src(
//    const NodePtr &node,
//    const std::vector<std::pair<int, ScalarType>> &arg_list,
//    std::string kernel_name) {
//  std::ostringstream oss;
//  genX86Src(node, arg_list, std::move(kernel_name), oss);
//  return oss.str();
//}
//
// std::string api::genX86src(
//    const NodePtr &node,
//    const std::vector<elena::Exchanger::ValueInfoPtr> &arg_hash_key_list,
//    const std::vector<elena::Exchanger::ValueInfoPtr> &input_hash_key_list,
//    std::string kernel_name, bool dyn_shape) {
//  int float_type = 0;
//  // if(arg_hash_key_list[0]->dtype == ir::ScalarType::Float16)
//  //   float_type = FP16;
//  // else if (arg_hash_key_list[0]->dtype == ir::ScalarType::BFloat16)
//  //   float_type = BF16;
//  auto s = api::genX86Header();
//
//  std::ofstream fout;
//  fout.open("./elena_int.h");
//  fout << s;
//  fout.close();
//
//  std::ostringstream oss;
//
//  int Float_type = 0;
//  // based on first value type
//  s = arg_hash_key_list[0]->dtype;
//  if (s == "wchar_t")
//    Float_type = 1;
//  else if (s == "uint16_t")
//    Float_type = 2;
//
//  // deal with empty function body
//  X86Codegen visitor{oss};
//  for (auto &arg : arg_hash_key_list)
//    visitor.sorter.markVisitedByName("Var" + arg->hash_key);
//
//  visitor << Prelude;
//  visitor << R"(extern "C" __global__ void )";
//  visitor << kernel_name << "(";
//
//  std::vector<std::string> arg_list_str;
//  for (int i = 0; i < arg_hash_key_list.size(); ++i) {
//    std::string str = arg_hash_key_list[i]->hash_key;
//    for (int j = 0; j < i; j++) {
//      if (str == arg_hash_key_list[j]->hash_key) {
//        str = str + "_copy";
//      }
//    }
//
//    // const char *type_str = SCALARTYPE_SYMBOL(arg_hash_key_list[i]->dtype);
//    std::string type_s = arg_hash_key_list[i]->dtype;  // type_str;
//    if (type_s == "wchar_t") {
//      type_s = "half";
//    }
//    if (type_s == "uint16_t") {
//      type_s = "nv_bfloat16";
//    }
//
//    // if (str == "bool") str = "float";
//    arg_list_str.push_back(type_s + "* " + str);
//  }
//
//  for (int i = 0; i < arg_list_str.size(); i++) {
//    visitor << arg_list_str[i];
//    if (i != arg_list_str.size() - 1) {
//      visitor << ", ";
//    }
//  }
//  visitor << ") " << block_begin;
//
//  if (node == nullptr) {
//    visitor << block_end;
//    return oss.str();
//  }
//  auto cur = node;
//  while (cur->get_type() == IRNodeType::Attr)
//    cur = ir::ptr_cast<Attr, Node>(cur)->body;
//  ELENA_ASSERT_EQ(cur->get_type(), IRNodeType::Allocate, "Node type
//  mismatch.");
//
//  auto anode = ir::ptr_cast<Allocate, Node>(cur);
//  // WARNING: This is an ugly patch, should be fixed elsewhere
//  // visitor << prelude;
//  // if the top allcate's var is not showed in function attrs, we still need
//  to
//  // allocate it
//  auto anode_var = anode->var;
//  auto anode_var_name = anode_var->get_name();
//  visitor.visit(anode->body);
//  visitor << block_end;
//
//  return oss.str();
//}
//
// std::string api::genX86Header() {
//  return R"(
// #include <math.h>
// #include <stdint.h>
//
// #define sqrf(x) (x * x)
// #define signf(x) (x > 0) - (x < 0)
// #define sign2f(x) (fmax((float)0, (float)signf(x)))
// #define reluf(x) fmax(x, 0)
// #define seluf(x)                       \
//   (1.0507009873554804934193349852946 * \
//    (x > 0 ? x : 1.6732632423543772848170429916717 * (exp(x) - 1)))
// #define sigmoidf(x) (1 / (1 + exp(-x)))
// #define remainder(x, y) ((x) - (y)*floor((x) / (y)))
// #define rrelu_bin(x, y) (x >= 0 ? x : x * y)
// #define rrelu_rand(x, y) (x + (y - x) * generateRandom(25))
// #define reach(x, y) (x >= y ? 1 : 0)
// #define beyond(x, y) (x > y ? 1 : 0)
// #define same(x, y) (x == y ? 1 : 0)
// #define rounds(x)                                                      \
//   ((int)(x)&1 ? roundf(x)                                              \
//               : (x >= 0 ? (x - floorf(x) > 0.5 ? ceilf(x) : floorf(x)) \
//                         : (ceilf(x) - x > 0.5 ? floorf(x) : ceilf(x))))
// #define reverse01(x, y) (x > 0 ? 1 : y)
// )";
// }
