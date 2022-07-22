//===-- elena/src/codegen/DeviceCodegen.cpp
// - Code generate for device kernel code -------*- C++ -*-===//
//
// Part of the Elena Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the DeviceCodegen
/// class, which is used to generate device kernel code.
///
//===----------------------------------------------------------------------===//

#include "CodeGen/DeviceCodegen.h"

template <typename DeviceType>
DeviceCodegen<DeviceType>::DeviceCodegen(std::ostream &output_stream,
                                         std::set<int> var_mentioned)
    : TextGen(output_stream), VarMentioned(std::move(var_mentioned)) {}

template <typename DeviceType>
DeviceCodegen<DeviceType>::DeviceCodegen(
    std::ostream &output_stream, std::set<std::string> str_var_mentioned)
    : TextGen(output_stream), StrVarMentioned(std::move(str_var_mentioned)) {}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Cast *cast_ptr) {
  *this << '(' << TYPE_OF(cast_ptr) << ')';
  *this << '(';
  visit(cast_ptr->expr_);
  *this << ')';
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::ScalarAssign *sa_ptr) {
    visit(sa_ptr->var);
    *this << " = ";
    visit(sa_ptr->value);
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::BroadcastSymbol *BS_ptr) {
  *this << "make_float" << BS_ptr->get_lanes() << "(";
  for (int i = 0; i < BS_ptr->get_lanes() - 1; i++) {
    visit(BS_ptr->base_);
    *this << ", ";
  }
  visit(BS_ptr->base_);
  *this << ")";
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Allocate *allocate_ptr) {
  CHECK_NODE_TYPE(allocate_ptr->var, TensorVar)
  auto tensor_ptr = ir::ptr_cast<TensorVar>(allocate_ptr->var);
  auto tensor_name = tensor_ptr->get_name();

  if (std::is_same<DeviceType, CCode>::value) {
    markVisited(tensor_ptr.get());
  }

  // tensorize allocate: print nvcuda::wmma::fragment intrinsic
  if (allocate_ptr->is_tensorize && allocate_ptr->args.size() >= 5 &&
      (std::is_same<DeviceType, CudaCode>::value ||
       std::is_same<DeviceType, TangCode>::value)) {
    // tensorize is for CUDA only currently
    *this << "nvcuda::wmma::fragment<nvcuda::wmma::";
    if (allocate_ptr->args[0] == 0) {
      *this << "matrix_a, ";
    } else if (allocate_ptr->args[0] == 1) {
      *this << "matrix_b, ";
    } else if (allocate_ptr->args[0] == 2) {
      *this << "accumulator, ";
    }
    *this << allocate_ptr->args[1];
    *this << ", ";
    *this << allocate_ptr->args[2];
    *this << ", ";
    *this << allocate_ptr->args[3];
    *this << ", ";
    if (allocate_ptr->args[4] == static_cast<int>(ir::ScalarType::Float16)) {
      *this << "half";
    } else if (allocate_ptr->args[4] ==
               static_cast<int>(ir::ScalarType::Float32)) {
      *this << "float";
    }
    if (allocate_ptr->args[0] < 2) {
      *this << ", nvcuda::wmma::row_major> ";
    } else {
      *this << "> ";
    }
    *this << tensor_name;
    for (const auto &rg : allocate_ptr->bound->element) {
      *this << "[";
      visit(rg->extent);
      *this << "]";
    }
    *this << ";" << endl;
    visit(allocate_ptr->body);
    return;
  }

  // copied to .share tensor cannot be assigned, ending with _copied means its
  // the first hand res of copied op which means it's the aim of this process
  if (tensor_name.length() > 7 &&
      tensor_name.substr(tensor_name.length() - 7, 7) == "_copied") {
    visit(allocate_ptr->body);
    return;
  }
  /* For str_var_mentioned condition */
  bool str_var_is_mentioned_now = false;
  /* For var_mentioned condition */
  bool var_is_mentioned_now = false;
  int mentioned_tensor_id = 0;

  if (StrVarMentioned.size() != 0) {
    ELENA_ASSERT(
        VarMentioned.size() == 0,
        "str_var_mentioned and var_mentioned cannot appear at the same time");

    bool is_output = false;
    // for (auto &hash_key : ArgHashKeyList) {
    //   if (tensor_name == hash_key->hash_key) {
    //     is_output = true;
    //   }
    // }

    if (is_output) {
      visit(allocate_ptr->body);
      return;
    } else if (StrVarMentioned.count(tensor_name) > 0) {
      // dont assign this tensor again cause its already mentioned in var in
      // cuda function other situation aren't thinked about for now
      visit(allocate_ptr->body);
      return;
    } else {
      str_var_is_mentioned_now = true;
    }
  } else if (tensor_name.length() > 3) {
    auto tensor_id_str = tensor_name.substr(3, tensor_name.length() - 3);
    auto tensor_id = std::strtol(tensor_id_str.c_str(), nullptr, 10);

    // bool is_output = std::any_of(
    //     ArgHashKeyList.begin(), ArgHashKeyList.end(),
    //     [&tensor_name](const elena::Exchanger::ValueInfoPtr &arg_hash_key) {
    //       return arg_hash_key->hash_key == tensor_name;
    //     });

    // if (is_output) {
    //   visit(allocate_ptr->body);
    //   return;
    // }

    auto is_all_numeric = [](const std::string &tensor_id_str) {
      return std::all_of(
          tensor_id_str.begin(), tensor_id_str.end(),
          [](const char &aim) { return aim > '0' && aim < '9'; });
    };

    if (is_all_numeric(tensor_id_str)) {
      if (VarMentioned.count(tensor_id) > 0) {
        visit(allocate_ptr->body);
        return;
      } else {
        var_is_mentioned_now = true;
        mentioned_tensor_id = tensor_id;
      }
    }
  }

  std::unordered_map<std::string, std::string> device_memory;

  if (std::is_same<DeviceType, CambriconCode>::value) {
    device_memory = CambriconMemory;
  } else {
    device_memory = CudaMemory;
  }

  auto it = device_memory.find(tensor_ptr->get_name());
  if (it != device_memory.end()) *this << it->second;
  *this << TYPE_OF(tensor_ptr) << " ";
  visit(tensor_ptr);

  for (const auto &rg : allocate_ptr->bound->element) {
    *this << "[";
    visit(rg->extent);
    *this << "]";
  }

  *this << ";" << endl;
  // do this to avoid duplicate assignment to var when theres an copy op to a
  // not .shared tensor
  if (str_var_is_mentioned_now) {
    StrVarMentioned.insert(tensor_name);
  } else if (var_is_mentioned_now) {
    VarMentioned.insert(mentioned_tensor_id);
  }
  visit(allocate_ptr->body);
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Provide *provide_ptr) {
  visit(provide_ptr->var);
  *this << "[";
  visit(provide_ptr->index);
  *this << "]";
  *this << " = ";
  visit(provide_ptr->value);
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::For *for_stmt_ptr) {
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

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Store *store_ptr) {
  if (store_ptr->index->element[0]->get_type() == ir::IRNodeType::Ramp) {
    auto ramp = ir::ptr_cast<ir::Ramp>(store_ptr->index->element[0]);
    if (store_ptr->var->get_name().substr(0, 6) == "share_") {
      *this << "((__shared__ float" << ramp->lanes << "*)(";
    } else {
      *this << "((float" << ramp->lanes << "*)(";
    }
    visit(store_ptr->var);
    *this << " + ";
    visit(ramp->base);
    *this << "))[0]";
  } else {
    visit(store_ptr->var);
    for (const auto &index : store_ptr->index->element) {
      *this << "[";
      visit(index);
      *this << "]";
    }
  }
  *this << " = ";
  visit(store_ptr->value);
  *this << ";" << endl;
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::TensorVar *tensor_ptr) {
  if (std::is_same<DeviceType, CCode>::value && ENABLE_SORTER) {
    if (WellDefined.find(tensor_ptr->get_name()) != WellDefined.cend()) return;
    markVisited(tensor_ptr);
    // recursively visit children
    visit(tensor_ptr->op);
    // define the tensor and initialize to zeros.
    *this << TYPE_OF(tensor_ptr) << " ";
    this->visit(tensor_ptr);
    for (const auto &n : tensor_ptr->shape->element) {
      *this << '[';
      this->visit(n);
      *this << ']';
    }
    *this << " = {0};" << endl;
    // expand the op to 'for' loops.
    this->visit(tensor_ptr->op);
  }
  *this << makeIdentifier(tensor_ptr->get_name());
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::IterVar *iter_ptr) {
  std::vector<std::string> reserved;
  if (std::is_same<DeviceType, CudaCode>::value ||
      std::is_same<DeviceType, TangCode>::value) {
    reserved = {"threadIdx.x", "threadIdx.y", "threadIdx.z",
                "blockIdx.x",  "blockIdx.y",  "blockIdx.z",
                "blockDim.x",  "blockDim.y",  "blockDim.z"};
  } else if (std::is_same<DeviceType, HipCode>::value) {
    reserved = {"hipThreadIdx_x", "hipThreadIdx_y", "hipThreadIdx_z",
                "hipBlockIdx_x",  "hipBlockIdx_y",  "hipBlockIdx_z",
                "hipBlockDim_x",  "hipBlockDim_y",  "hipBlockDim_z"};
  } else if (std::is_same<DeviceType, CambriconCode>::value) {
    reserved = {"taskId"};
  }

  const auto &name = iter_ptr->get_name();
  if (std::any_of(reserved.begin(), reserved.end(),
                  [name](const std::string &s) { return name == s; })) {
    *this << name;
  } else {
    *this << makeIdentifier(name);
  }
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::ScalarVar *scalar_ptr) {
  if (!scalar_ptr->tensor) {
    *this << scalar_ptr->get_name();
  } else if (!scalar_ptr->is_placeholder()) {
    if (scalar_ptr->indices->element[0]->get_type() == ir::IRNodeType::Ramp) {
      auto ramp = ir::ptr_cast<ir::Ramp>(scalar_ptr->indices->element[0]);
      if (scalar_ptr->tensor->get_name().substr(0, 6) == "share_") {
        *this << "((__shared__ float" << ramp->lanes << "*)(";
      } else {
        *this << "((float" << ramp->lanes << "*)(";
      }
      visit(scalar_ptr->tensor);
      *this << " + ";
      visit(ramp->base);
      *this << "))[0]";
    } else {
      visit(scalar_ptr->tensor);
      for (const auto &index : scalar_ptr->indices->element) {
        *this << "[";
        visit(index);
        *this << "]";
      }
    }
  } else {
    ELENA_ABORT("There might be a bug here")
  }
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Attr *attr_ptr) {
  if (attr_ptr->key == AttrType::StorageScope ||
      attr_ptr->key == AttrType::RealizeScope) {
    CHECK_NODE_TYPE(attr_ptr->value, Label)
    auto label = ir::ptr_cast<Label>(attr_ptr->value);
    if (auto p = ir::ptr_cast<ir::TensorVar>(attr_ptr->node)) {
      if (std::is_same<DeviceType, CambriconCode>::value) {
        CambriconMemory.emplace(p->get_name(), label->get_value() + " ");
      } else if (label->get_value() == "__shared__") {
        // we only use "__shared__" key word in Cuda or Hip
        CudaMemory.emplace(p->get_name(), label->get_value() + " ");
      }
    }
  }

  bool VisitThreadExtent = attr_ptr->key == AttrType::ThreadExtent &&
                           std::is_same<DeviceType, CCode>::value;
  if (VisitThreadExtent) {
    // expand a threadIdx/blockIdx into a loop for C code
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
  if (VisitThreadExtent) {
    *this << block_end;
  }
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Logical *logical_ptr) {
  *this << "(";
  visit(logical_ptr->lhs);
  *this << " " << LOGICALTYPE_SYMBOL(logical_ptr->operation_type) << " ";
  visit(logical_ptr->rhs);
  *this << ")";
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Unary *unary_ptr) {
  const auto type = unary_ptr->operand->get_dtype();
  if (std::is_same<DeviceType, CCode>::value) {
    if (type == ir::ScalarType::Float32 &&
        unary_ptr->operation_type == UnaryType::Abs) {
      *this << "(fabs(";
    } else {
      *this << "(" << UOP_DEVICE_NAME(unary_ptr->operation_type) << "(";
    }
    visit(unary_ptr->operand);
    *this << "))";
    return;
  }

  if (unary_ptr->operation_type == UnaryType::Abs) {
    // todo: @qianchao refactor
    if (type == ir::ScalarType::Float16 || type == ir::ScalarType::BFloat16) {
      *this << "("
            << "__habs(";
    } else if (type == ir::ScalarType::Float32) {
      *this << "("
            << "fabs(";
    } else {
      *this << "(" << UOP_DEVICE_NAME(unary_ptr->operation_type) << "(";
    }
    visit(unary_ptr->operand);
    *this << "))";
  } else if (unary_ptr->operation_type == UnaryType::Negate) {
    // todo: UInt8
    if (type == ir::ScalarType::UInt32 || type == ir::ScalarType::UInt8) {
      *this << "(-"
            << "static_cast<int>(";
      visit(unary_ptr->operand);
      *this << "))";
    } else if (type == ir::ScalarType::UInt64) {
      *this << "(-"
            << "static_cast<int64_t>(";
      visit(unary_ptr->operand);
      *this << "))";
    } else {
      *this << "(-"
            << "(";
      visit(unary_ptr->operand);
      *this << "))";
    }
  } else if (unary_ptr->operation_type == UnaryType::Floor) {
    // todo: @qianchao refactor
    if (type == ir::ScalarType::Float16 || type == ir::ScalarType::BFloat16) {
      *this << "("
            << "hfloor(";
    } else if (type == ir::ScalarType::Float64) {
      *this << "("
            << "floor(";
    } else if (type == ir::ScalarType::UInt64) {
      *this << "(static_cast<uint64_t>(floorf(";
    } else {
      *this << "(" << UOP_DEVICE_NAME_F(unary_ptr->operation_type) << "(";
    }
    visit(unary_ptr->operand);
    *this << "))";
    if (type == ir::ScalarType::UInt64) {
      *this << ")";
    }
  } else if (unary_ptr->operation_type == UnaryType::Cround) {
    *this << "roundf(";
    visit(unary_ptr->operand);
    *this << ")";
  } else if (unary_ptr->operation_type == UnaryType::Round) {
    *this << "rounds(";
    visit(unary_ptr->operand);
    *this << ")";
  } else if (unary_ptr->operation_type == UnaryType::IsInf) {
    if (type == ir::ScalarType::Float16 || type == ir::ScalarType::BFloat16) {
      *this << "__hisinf(";
    } else {
      *this << "isinf(";
    }
    visit(unary_ptr->operand);
    *this << ")";
  } else if (unary_ptr->operation_type == UnaryType::IsNan) {
    *this << "isnan(";
    visit(unary_ptr->operand);
    *this << ")";
  } else if (unary_ptr->operation_type == UnaryType::Ceil) {
    if (type == ir::ScalarType::Float16 || type == ir::ScalarType::BFloat16) {
      *this << "(" << H_UOP_DEVICE_NAME(unary_ptr->operation_type) << "(";
    } else if (type == ir::ScalarType::Float32) {
      *this << "(" << UOP_DEVICE_NAME_F(unary_ptr->operation_type) << "(";
    } else if (type == ir::ScalarType::UInt64) {
      *this << "(" << "ceil_uint64" << "(";
    } else {
      *this << "(" << UOP_DEVICE_NAME(unary_ptr->operation_type) << "(";
    }

    if (unary_ptr->operand->get_type() == ir::IRNodeType::Binary) {
      auto ceil_binary_ptr = ir::ptr_cast<ir::Binary>(unary_ptr->operand);
      if (ceil_binary_ptr->operation_type == ir::BinaryType::Div) {
        *this << "(";
        if (type == ScalarType::Float16) {
          *this << "(half)";
        } else if (type == ScalarType::BFloat16) {
          *this << "(nv_bfloat16)";
        } else {
          *this << "(float)";
        }
        *this << "(";
        visit(ceil_binary_ptr->lhs);
        *this << ")";
        *this << " " << BINARYTYPE_SYMBOL(ceil_binary_ptr->operation_type)
              << " ";
        if (type == ScalarType::Float16) {
          *this << "(half)";
        } else if (type == ScalarType::BFloat16) {
          *this << "(nv_bfloat16)";
        }
        *this << "(";
        visit(ceil_binary_ptr->rhs);
        *this << ")";
        *this << ")";
        *this << "))";
      } else {
        visit(unary_ptr->operand);
        *this << "))";
      }
    } else {
      visit(unary_ptr->operand);
      *this << "))";
    }
  } else {
    if (type == ir::ScalarType::Float16 || type == ir::ScalarType::BFloat16) {
      *this << "(" << H_UOP_DEVICE_NAME(unary_ptr->operation_type) << "(";
    } else if (type == ir::ScalarType::Float32) {
      *this << "(" << UOP_DEVICE_NAME_F(unary_ptr->operation_type) << "(";
    } else {
      *this << "(" << UOP_DEVICE_NAME(unary_ptr->operation_type) << "(";
    }
    visit(unary_ptr->operand);
    *this << "))";
  }
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Binary *binary_ptr) {
  const auto type = binary_ptr->get_dtype();
  if (binary_ptr->operation_type == BinaryType::Mod) {
    if (type == ScalarType::Float32 || type == ScalarType::Float64) {
      *this << "remainder((float)";
      visit(binary_ptr->lhs);
      *this << ", (float)";
      visit(binary_ptr->rhs);
      *this << ")";
    } else {
      *this << "(";
      visit(binary_ptr->lhs);
      *this << " % ";
      visit(binary_ptr->rhs);
      *this << ")";
    }
  } else if (binary_ptr->operation_type == BinaryType::Pow) {
    if (type == ScalarType::Float16 || type == ScalarType::BFloat16) {
      *this << "hpow(";
    } else {
      *this << "powf(";
    }
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::Max) {
    if (type == ScalarType::UInt64) {
      *this << "max((uint64_t)";
      visit(binary_ptr->lhs);
      *this << ", (uint64_t)";
      visit(binary_ptr->rhs);
      *this << ")";
    } else if (type == ScalarType::Float16) {
      *this << "max((half)";
      visit(binary_ptr->lhs);
      *this << ", (half)";
      visit(binary_ptr->rhs);
      *this << ")";
    } else if (type == ScalarType::BFloat16) {
      *this << "max((nv_bfloat16)";
      visit(binary_ptr->lhs);
      *this << ", (nv_bfloat16)";
      visit(binary_ptr->rhs);
      *this << ")";
    } else {
      if (std::is_same<DeviceType, CudaCode>::value ||
          std::is_same<DeviceType, TangCode>::value) {
        *this << "fmax((float)";
        visit(binary_ptr->lhs);
        *this << ", (float)";
      } else {
        *this << "max(";
        visit(binary_ptr->lhs);
        *this << ", ";
      }
      visit(binary_ptr->rhs);
      *this << ")";
    }
  } else if (binary_ptr->operation_type == BinaryType::Min) {
    if (type == ScalarType::UInt64) {
      *this << "min((uint64_t)";
      visit(binary_ptr->lhs);
      *this << ", (uint64_t)";
      visit(binary_ptr->rhs);
      *this << ")";
    } else if (type == ScalarType::Float16) {
      *this << "min((half)";
      visit(binary_ptr->lhs);
      *this << ", (half)";
      visit(binary_ptr->rhs);
      *this << ")";
    } else if (type == ScalarType::BFloat16) {
      *this << "min((nv_bfloat16)";
      visit(binary_ptr->lhs);
      *this << ", (nv_bfloat16)";
      visit(binary_ptr->rhs);
      *this << ")";
    } else {
      if (std::is_same<DeviceType, CudaCode>::value ||
          std::is_same<DeviceType, TangCode>::value) {
        *this << "fmin((float)";
        visit(binary_ptr->lhs);
        *this << ", (float)";
      } else {
        *this << "min(";
        visit(binary_ptr->lhs);
        *this << ", ";
      }
      visit(binary_ptr->rhs);
      *this << ")";
    }
  } else if (binary_ptr->operation_type == BinaryType::Norm) {
    *this << "normf(";
    visit(binary_ptr->rhs);
    *this << ", &";
    visit(binary_ptr->lhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::EitherOr) {
    auto zero_or_one = binary_ptr->lhs;
    auto binary_expr = binary_ptr->rhs;
    auto left_right_val = ir::ptr_cast<ir::Binary>(binary_expr);
    auto left_val = left_right_val->lhs;
    auto right_val = left_right_val->rhs;
    *this << "either_or(";
    visit(zero_or_one);
    *this << ", ";
    visit(left_val);
    *this << ", ";
    visit(right_val);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::MulNan2Zero) {
    *this << "mul_nan2zero(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::RreluBin) {
    *this << "rrelu_bin(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::RreluRand) {
    *this << "rrelu_rand(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::Reverse01) {
    *this << "reverse01(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::Reach) {
    *this << "reach(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::Beyond) {
    *this << "beyond(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::Same) {
    *this << "same(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::SignMul) {
    *this << "sign_mul(";
    visit(binary_ptr->lhs);
    *this << ", ";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::Sll) {
    *this << "((int)";
    visit(binary_ptr->lhs);
    *this << " << (int)";
    visit(binary_ptr->rhs);
    *this << ")";
  } else if (binary_ptr->operation_type == BinaryType::Slr) {
    *this << "((int)";
    visit(binary_ptr->lhs);
    *this << " >> (int)";
    visit(binary_ptr->rhs);
    *this << ")";
  } else {
    *this << "(";
    if (type == ScalarType::Float16) {
      *this << "(half)";
    } else if (type == ScalarType::BFloat16) {
      *this << "(nv_bfloat16)";
    }
    *this << "(";
    visit(binary_ptr->lhs);
    *this << ")";
    *this << " " << BINARYTYPE_SYMBOL(binary_ptr->operation_type) << " ";
    if (type == ScalarType::Float16) {
      *this << "(half)";
    } else if (type == ScalarType::BFloat16) {
      *this << "(nv_bfloat16)";
    }
    *this << "(";
    visit(binary_ptr->rhs);
    *this << ")";
    *this << ")";
  }
}

template <typename DeviceType>
template <typename T>
void DeviceCodegen<DeviceType>::visit(ir::Const<T> *const_ptr) {
  if (std::is_same<DeviceType, CCode>::value) {
    *this << std::boolalpha << const_ptr->get_value();
    return;
  }

  if (const_ptr->get_dtype() == ir::ScalarType::Float32) {
    // Consider the condition where
    // the two operands have no mantissa.
    *this << "(static_cast<float>(" << const_ptr->get_value() << "))";
  } else if (const_ptr->get_dtype() == ir::ScalarType::Float64) {
    *this << "(static_cast<double>(" << const_ptr->get_value() << "))";
  } else {
    *this << std::boolalpha << const_ptr->get_value();
  }
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Let *let_ptr) {
  // DISCUSS: Possible incorrect scope
  // Should we remove the explicit block_begin/block_end and
  // rely on that the user should not introduce a name clash?
  // *this << block_begin;
  *this << "const " << TYPE_OF(let_ptr->var) << " ";
  visit(let_ptr->var);
  *this << " = ";
  visit(let_ptr->value);
  *this << ";" << endl;
  visit(let_ptr->body);
  // *this << block_end;
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::IfThenElse *if_then_else_ptr) {
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

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(Select *select_ptr) {
  *this << "(";
  visit(select_ptr->cond);
  *this << " ? ";
  visit(select_ptr->tBranch);
  *this << " : ";
  visit(select_ptr->fBranch);
  *this << ")";
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Call *call_ptr) {
  if (call_ptr->func == CallFunction::Sync) {
    *this << "__syncthreads();" << endl;
  } else if (call_ptr->func == CallFunction::atomic_add &&
             call_ptr->args->element.size() == 2) {
    *this << "atomicAdd(&";
    visit(call_ptr->args->element[0]);
    *this << ",";
    visit(call_ptr->args->element[1]);
    *this << ");";
  } else if (call_ptr->func == CallFunction::atomic_max &&
             call_ptr->args->element.size() == 2) {
    if (call_ptr->get_dtype() == ir::ScalarType::Float32) {
      (*this).AtomicType = 1;
      *this << "atomicMaxFloat(&";
    } else if (call_ptr->get_dtype() == ir::ScalarType::Float64) {
      (*this).AtomicType = 2;
      *this << "atomicMaxDouble(&";
    } else {
      *this << "atomicMax(&";
    }
    visit(call_ptr->args->element[0]);
    *this << ",";
    visit(call_ptr->args->element[1]);
    *this << ");";
  } else if (call_ptr->func == CallFunction::atomic_min &&
             call_ptr->args->element.size() == 2) {
    if (call_ptr->get_dtype() == ir::ScalarType::Float32) {
      (*this).AtomicType = 1;
      *this << "atomicMinFloat(&";
    } else if (call_ptr->get_dtype() == ir::ScalarType::Float64) {
      (*this).AtomicType = 2;
      *this << "atomicMinDouble(&";
    } else {
      *this << "atomicMin(&";
    }
    visit(call_ptr->args->element[0]);
    *this << ",";
    visit(call_ptr->args->element[1]);
    *this << ");";
  } else if (call_ptr->func == CallFunction::wmma_fill_fragment &&
             call_ptr->args->element.size() >= 6) {
    *this << "(void)nvcuda::wmma::fill_fragment(";
    visit(call_ptr->args->element[0]);
    *this << "[";
    visit(call_ptr->args->element[4]);
    *this << "], ";
    visit(call_ptr->args->element[5]);
    *this << ");";
  } else if (call_ptr->func == CallFunction::wmma_load_matrix_sync &&
             call_ptr->args->element.size() >= 5) {
    *this << "(void)nvcuda::wmma::load_matrix_sync(";
    visit(call_ptr->args->element[0]);
    *this << "[";
    visit(call_ptr->args->element[1]);
    *this << "], ((half*)";
    visit(call_ptr->args->element[2]);
    *this << " + ";
    visit(call_ptr->args->element[3]);
    *this << "), ";
    visit(call_ptr->args->element[4]);
    *this << ");";
  } else if (call_ptr->func == CallFunction::wmma_mma_sync &&
             call_ptr->args->element.size() >= 8) {
    *this << "(void)nvcuda::wmma::mma_sync(";
    visit(call_ptr->args->element[0]);
    *this << "[";
    visit(call_ptr->args->element[1]);
    *this << "], ";
    visit(call_ptr->args->element[2]);
    *this << "[";
    visit(call_ptr->args->element[3]);
    *this << "], ";
    visit(call_ptr->args->element[4]);
    *this << "[";
    visit(call_ptr->args->element[5]);
    *this << "], ";
    visit(call_ptr->args->element[6]);
    *this << "[";
    visit(call_ptr->args->element[7]);
    *this << "]);";
  } else if (call_ptr->func == CallFunction::wmma_store_matrix_sync &&
             call_ptr->args->element.size() >= 5) {
    *this << "(void)nvcuda::wmma::store_matrix_sync(";
    *this << "((float*)";
    visit(call_ptr->args->element[0]);
    *this << " + ";
    visit(call_ptr->args->element[1]);
    *this << "), ";
    visit(call_ptr->args->element[2]);
    *this << "[";
    visit(call_ptr->args->element[3]);
    *this << "], ";
    visit(call_ptr->args->element[4]);
    *this << ", nvcuda::wmma::mem_row_major);";
  } else {
    throw std::runtime_error(
        "Calling function other than 'Sync' is not supported!");
  }
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ir::Evaluate *node) {
  visit(node->value);
  *this << endl;
}

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(ComputeOp *compute) {
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

template <typename DeviceType>
void DeviceCodegen<DeviceType>::visit(Reduce *reduce) {
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

template <typename DeviceType>
static inline void genSrcStream(const NodePtr &node,
                                const std::string &kernel_name,
                                DeviceCodegen<DeviceType> visitor,
                                const std::function<void()> &visit_arg_list,
                                std::set<int> var_mentioned = {}) {
  if (visitor.FloatType == 1) {
    visitor << "#include <cuda_fp16.h>\n";
    visitor << "__device__ half max"
            << "(half a, half b)\n"
            << "{\n  return __hgt(__half(a), __half(b)) ? a : b;\n}\n";
    visitor << "__device__ half min(half a, half b)\n"
            << "{\n  return __hlt(__half(a), __half(b)) ? a : b;\n}\n";
    visitor << CudaHalfUtil;
  } else if (visitor.FloatType == 2) {
    visitor << "#include <cuda_bf16.h>\n";
    visitor << "__device__ nv_bfloat16 max"
            << "(nv_bfloat16 a, nv_bfloat16 b)\n"
            << "{\n  return __hgt(a, b) ? a : b;\n}\n";
    visitor << "__device__ nv_bfloat16 min(nv_bfloat16 a, nv_bfloat16 b)\n"
            << "{\n  return __hlt(a, b) ? a : b;\n}\n";
    visitor << CudaBF16Util;
  }

  // visitor << Prelude;
  if (std::is_same<DeviceType, TangCode>::value) {
    visitor << PreludeTang;
  }

  if (std::is_same<DeviceType, CCode>::value) {
    visitor << R"(extern "C" void )";
  } else {
    visitor << R"(extern "C" __global__ void )";
  }
  visitor << kernel_name << "(";

  visit_arg_list();

  visitor << ") " << block_begin;

  if (node == nullptr) {
    visitor << block_end;
    return;
  }
  auto cur = node;
  while (cur->get_type() == IRNodeType::Attr)
    cur = ir::ptr_cast<Attr, Node>(cur)->body;
  ELENA_ASSERT_EQ(cur->get_type(), IRNodeType::Allocate, "Node type mismatch.")

  auto anode = ir::ptr_cast<Allocate, Node>(cur);

  // WARNING: This is an ugly patch, should be fixed elsewhere
  // visitor << Prelude;
  // if the top allcate's var is not showed in function attrs, we still need to
  // allocate it
  auto anode_var = anode->var;
  auto anode_var_name = anode_var->get_name();
  bool skip_allocate = false;
  assert(anode_var_name.length() > 3);
  auto numeric_part = anode_var_name.substr(3, anode_var_name.length() - 3);
  bool all_numeric =
      std::all_of(numeric_part.begin(), numeric_part.end(),
                  [](const char &i) { return i < '9' && i > '0'; });
  if (all_numeric) {
    auto var_id = std::strtol(numeric_part.c_str(), nullptr, 10);
    skip_allocate = var_mentioned.find(var_id) != var_mentioned.end();
  }
  if (skip_allocate) {
    visitor.visit(anode->body);
  } else {
    visitor.visit(anode);
  }
  visitor << block_end;

  std::ofstream fout;
  fout.open("./elena_int.h", std::ios_base::app);
  if (visitor.AtomicType == 1) {
    fout << atomicFloat;
  } else if (visitor.AtomicType == 2) {
    fout << atomicDouble;
  }
  fout.close();
}

template <typename DeviceType>
static inline void genSrcStream(const NodePtr &node,
                   const std::string &kernel_name,
                   DeviceCodegen<DeviceType> visitor,
                   const std::function<void()> &visit_arg_list,
                   std::set<std::string> str_var_mentioned = {},
                   const std::vector<ir::ScalarVarPtr> &vars = {}) {
  if (visitor.FloatType == 1) {
    visitor << "#include <cuda_fp16.h>\n";
    visitor << "__device__ half max"
            << "(half a, half b)\n"
            << "{\n  return __hgt(__half(a), __half(b)) ? a : b;\n}\n";
    visitor << "__device__ half min(half a, half b)\n"
            << "{\n  return __hlt(__half(a), __half(b)) ? a : b;\n}\n";
    visitor << CudaHalfUtil;
  } else if (visitor.FloatType == 2) {
    visitor << "#include <cuda_bf16.h>\n";
    visitor << "__device__ nv_bfloat16 max"
            << "(nv_bfloat16 a, nv_bfloat16 b)\n"
            << "{\n  return __hgt(a, b) ? a : b;\n}\n";
    visitor << "__device__ nv_bfloat16 min(nv_bfloat16 a, nv_bfloat16 b)\n"
            << "{\n  return __hlt(a, b) ? a : b;\n}\n";
    visitor << CudaBF16Util;
  }

  // visitor << Prelude;
  if (std::is_same<DeviceType, CCode>::value) {
    visitor << "\n" << R"(extern "C" void )";
  } else {
    visitor << "\n" << R"(extern "C" __global__ void )";
  }
  visitor << kernel_name << "(";

  visit_arg_list();

  visitor << ") " << block_begin;

  for (auto &var : vars) {
      visitor << "uint64_t " + var->get_name() << ";\n";
  }

  if (node == nullptr) {
    visitor << block_end;
    return;
  }
  auto cur = node;
  while (cur->get_type() == IRNodeType::Attr)
    cur = ir::ptr_cast<Attr, Node>(cur)->body;
  ELENA_ASSERT_EQ(cur->get_type(), IRNodeType::Allocate, "Node type mismatch.")

  auto anode = ir::ptr_cast<Allocate, Node>(cur);

  // WARNING: This is an ugly patch, should be fixed elsewhere
  // visitor << Prelude;
  // if the top allcate's var is not showed in function attrs, we still need to
  // allocate it
  auto anode_var = anode->var;
  auto anode_var_name = anode_var->get_name();
  bool skip_allocate = false;

  if (find(str_var_mentioned.begin(), str_var_mentioned.end(),
           anode_var_name) != str_var_mentioned.end()) {
    // not found
    skip_allocate = true;
  }
  if (skip_allocate) {
    visitor.visit(anode->body);
  } else {
    visitor.visit(anode);
  }
  visitor << block_end;

  std::ofstream fout;
  fout.open("./elena_int.h", std::ios_base::app);
  if (visitor.AtomicType == 1) {
    fout << atomicFloat;
  } else if (visitor.AtomicType == 2) {
    fout << atomicDouble;
  }
  fout.close();
}

std::string api::genCudaHeader(int float_type, const std::string &hname) {
  std::ostringstream oss;
  std::string header;
  oss << CudaHeader;
  if (float_type == FLOAT) {
    oss << FloatHeader;
  } else if (float_type == FP16) {
    oss << Fp16Header;
    oss << Common16BitHeader;
  } else if (float_type == BF16) {
    oss << BF16Header;
    oss << Common16BitHeader;
  }
  return oss.str();
}

std::string api::genTangHeader() {
  std::ostringstream oss;
  std::string header;
  oss << TangHeader;
  return oss.str();
}

std::string genX86Header() {
  return R"(
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#define min(a,b) ((a)>(b)?(b):(a))
#define max(a,b) ((a)>(b)?(a):(b))
#define sqrf(x) (x * x)
#define signf(x) (x > 0) - (x < 0)
#define sign2f(x) (fmax((float)0, (float)signf(x)))
#define reluf(x) fmax(x, 0)
#define seluf(x)                       \
  (1.0507009873554804934193349852946 * \
   (x > 0 ? x : 1.6732632423543772848170429916717 * (exp(x) - 1)))
#define sigmoidf(x) (1 / (1 + exp(-x)))
#define remainder(x, y) ((x) - (y)*floor((x) / (y)))
#define rrelu_bin(x, y) (x >= 0 ? x : x * y)
#define rrelu_rand(x, y) (x + (y - x) * generateRandom(25))
#define reach(x, y) (x >= y ? 1 : 0)
#define beyond(x, y) (x > y ? 1 : 0)
#define same(x, y) (x == y ? 1 : 0)
#define rounds(x)                                                      \
  ((int)(x)&1 ? roundf(x)                                              \
              : (x >= 0 ? (x - floorf(x) > 0.5 ? ceilf(x) : floorf(x)) \
                        : (ceilf(x) - x > 0.5 ? floorf(x) : ceilf(x))))
#define reverse01(x, y) (x > 0 ? 1 : y)
)";
}

template <typename DeviceType>
static inline std::string genDeviceSrc(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name) {
  int float_type = FLOAT;
  if (arg_list[0].second == ir::ScalarType::Float16)
    float_type = FP16;
  else if (arg_list[0].second == ir::ScalarType::BFloat16)
    float_type = BF16;
  std::string s;
  if (std::is_same<DeviceType, CCode>::value) {
    s = genX86Header();
  } else {
    s = api::genCudaHeader(float_type);
  }
  std::ofstream fout;
  fout.open("./elena_int.h");
  fout << s;
  fout.close();

  if (std::is_same<DeviceType, TangCode>::value) {
    s = api::genTangHeader();
    std::ofstream fout;
    fout.open("./tang.h");
    fout << s;
    fout.close();
  }

  std::ostringstream oss;
  std::set<int> var_mentioned;
  const char *type_str = SCALARTYPE_SYMBOL(arg_list[0].second);
  int Float_type = strcmp(type_str, "wchar_t") == 0
                       ? 1
                       : (strcmp(type_str, "uint16_t") == 0 ? 2 : 0);
  for (const auto &i : arg_list) {
    var_mentioned.insert(i.first);
  }

  DeviceCodegen<DeviceType> visitor{oss, var_mentioned};
  visitor.FloatType = Float_type;

  auto visit_arg_list = [&]() {
    auto visit_arg = [&](size_t i) {
      const char *str = SCALARTYPE_SYMBOL(arg_list[i].second);
      str = strcmp(str, "wchar_t") == 0
                ? "half"
                : (strcmp(str, "uint16_t") == 0 ? "nv_bfloat16" : str);
      visitor << str << "* __restrict__ Var" << arg_list[i].first;
    };

    for (int i = 0; i < arg_list.size() - 1; ++i) {
      visit_arg(i);
      visitor << ", ";
    }
    visit_arg(arg_list.size() - 1);
  };

  genSrcStream<DeviceType>(node, kernel_name, visitor, visit_arg_list,
                           var_mentioned);
  return oss.str();
}

template <typename DeviceType>
static inline std::string genDeviceSrc(const ir::NodePtr &node,
                          const std::vector<ir::ExprPtr> &arg_list,
                          const std::string &kernel_name,
                          const std::vector<ir::ScalarVarPtr> &varlist = {}) {
  int float_type = FLOAT;
  if (arg_list[0]->get_dtype() == ir::ScalarType::Float16)
    float_type = FP16;
  else if (arg_list[0]->get_dtype() == ir::ScalarType::BFloat16)
    float_type = BF16;

  std::string s;
  if (std::is_same<DeviceType, CCode>::value) {
    s = genX86Header();
  } else {
    s = api::genCudaHeader(float_type);
  }

  std::ofstream fout;
  fout.open("./elena_int.h");
  fout << s;
  fout.close();

  if (std::is_same<DeviceType, TangCode>::value) {
    s = api::genTangHeader();
    std::ofstream fout;
    fout.open("./tang.h");
    fout << s;
    fout.close();
  }

  std::ostringstream oss;
  std::set<std::string> str_var_mentioned;
  const char *type_str = SCALARTYPE_SYMBOL(arg_list[0]->get_dtype());
  int Float_type = strcmp(type_str, "wchar_t") == 0
                       ? 1
                       : (strcmp(type_str, "uint16_t") == 0 ? 2 : 0);
  for (const auto &i : arg_list) {
    if (i->get_type() == ir::IRNodeType::TensorVar) {
      auto var = ptr_cast<TensorVar>(i);
      str_var_mentioned.insert(var->get_name());
    } else if (i->get_type() == ir::IRNodeType::ScalarVar) {
      auto var = ptr_cast<ScalarVar>(i);
      str_var_mentioned.insert(var->get_name());
    } else {
      ELENA_ABORT("arg_list not support this type");
    }
  }

  DeviceCodegen<DeviceType> visitor{oss, str_var_mentioned};
  visitor.FloatType = Float_type;

  auto visit_arg_list = [&]() {
    auto visit_arg = [&](size_t i) {
      const char *str = SCALARTYPE_SYMBOL(arg_list[i]->get_dtype());
      str = strcmp(str, "wchar_t") == 0
                ? "half"
                : (strcmp(str, "uint16_t") == 0 ? "nv_bfloat16" : str);
      if (arg_list[i]->get_type() == ir::IRNodeType::TensorVar) {
        auto var = ptr_cast<TensorVar>(arg_list[i]);
        visitor << str << "* __restrict__ " << var->get_name();
      } else if (arg_list[i]->get_type() == ir::IRNodeType::ScalarVar) {
        auto var = ptr_cast<ScalarVar>(arg_list[i]);
        visitor << str << " " << var->get_name();
      }
    };

    for (int i = 0; i < arg_list.size() - 1; ++i) {
      visit_arg(i);
      visitor << ", ";
    }
    visit_arg(arg_list.size() - 1);
  };

  genSrcStream<DeviceType>(node, kernel_name, visitor, visit_arg_list,
                           str_var_mentioned, varlist);
  return oss.str();
}

std::string api::genX86Raw(const TensorVarPtr &var,
                           const std::vector<TensorVarPtr> &args) {
  std::ostringstream oss;
  DeviceCodegen<CCode> visitor{oss, std::set<int>{}};
  for (auto &x : args) visitor.markVisited(x.get());

  visitor << "void kernel_";
  visitor.visit(var);
  visitor << "(";

  // use a "n + 1/2" style loop
  // or rather "for-each-but-last" loop
  auto x = begin(args);
  while (true) {
    std::string str = TYPE_OF(*x);
    // copied from the patch above, a REALLY wierd patch though
    if (str == "bool") str = "float";
    visitor << str << " __restrict__ ";
    visitor.visit(*x);
    for (auto &n : (*x)->shape->element) {
      visitor << '[';
      visitor.visit(n);
      visitor << ']';
    }
    if (++x == end(args)) break;
    visitor << ", ";
  }

  visitor << ") " << block_begin;
  visitor.visit(var->op);
  visitor.visit(var->op);
  visitor << block_end;
  return oss.str();
}

// todo: Integrate the interfaces together, in Elena v0.3+.
std::string api::genCudaSrc(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name) {
  return genDeviceSrc<CudaCode>(node, arg_list, kernel_name);
}

std::string api::genCudaSrc(
    const ir::NodePtr &node,
    const std::vector<ir::ExprPtr> &arg_list,
    const std::string &kernel_name,
    const std::vector<ir::ScalarVarPtr> &varlist) {
  return genDeviceSrc<CudaCode>(node, arg_list, kernel_name, varlist);
}

std::string api::genTangSrc(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name) {
  return genDeviceSrc<TangCode>(node, arg_list, kernel_name);
}

std::string api::genTangSrc(
    const ir::NodePtr &node,
    const std::vector<ir::ExprPtr> &arg_list,
    const std::string &kernel_name) {
  return genDeviceSrc<TangCode>(node, arg_list, kernel_name);
}

std::string api::genHipSrc(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name) {
  return genDeviceSrc<HipCode>(node, arg_list, kernel_name);
}

std::string api::genBangSrc(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name) {
  return genDeviceSrc<HipCode>(node, arg_list, kernel_name);
}

std::string api::genX86Src(
    const ir::NodePtr &node,
    const std::vector<std::pair<int, ir::ScalarType>> &arg_list,
    const std::string &kernel_name) {
  return genDeviceSrc<CCode>(node, arg_list, kernel_name);
}

std::string api::genX86Src(
    const ir::NodePtr &node,
    const std::vector<ir::ExprPtr> &arg_list,
    const std::string &kernel_name,
    const std::vector<ir::ScalarVarPtr> &varlist) {
  return genDeviceSrc<CCode>(node, arg_list, kernel_name, varlist);
}
