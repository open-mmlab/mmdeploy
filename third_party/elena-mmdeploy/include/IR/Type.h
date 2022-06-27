#ifndef ELENA_INCLUDE_IR_TYPE_H_
#define ELENA_INCLUDE_IR_TYPE_H_

#include <cstdint>
#include <string>
#include <unordered_map>

#include "logging.h"

namespace ir {

/**
 * @brief Node type(Expr,Stmt,Array)
 * @author hanruobing
 */
enum class IRNodeType : int {
#define IR_NODE_TYPE(type) type,
#include "x/ir_node_types.def"
};

/**
 * @brief Check node type, assert on type mismatch.
 * @author xupengcheng
 */
#define CHECK_NODE_TYPE(node, type)                         \
  ELENA_ASSERT_EQ(node->get_type(), ::ir::IRNodeType::type, \
                  "Node type unexpected");

/**
 * @brief Printable names for IRNodeType values.
 * @author xupengcheng
 */
static const char *IRNodeTypeName[] = {
#define IR_NODE_TYPE(type) #type,
#include "x/ir_node_types.def"
};
#define IRNODETYPE_NAME(x) (::ir::IRNodeTypeName[static_cast<int>(x)])

/**
 * @brief Call functions type
 * @author guanzhichao
 */
enum class CallFunction : int {
#define TYPE_CALL_FUNCTIONS(type) type,
#include "x/call_function_types.def"
};

static const char *CallFunctionName[] = {
#define TYPE_CALL_FUNCTIONS(type) #type,
#include "x/call_function_types.def"
};
#define CALL_FUNCTION_NAME(x) (::ir::CallFunctionName[static_cast<int>(x)])

/**
 * @brief Type for Binary node
 * @author guanzhichao
 */
enum class BinaryType : int {
#define TYPE_BINARYTYPE_OP_MAP(opname, op) opname,
#include "x/binary_types.def"
};

static const char *BinaryTypeSymbol[] = {
#define TYPE_BINARYTYPE_OP_MAP(opname, op) #op,
#include "x/binary_types.def"
};
#define BINARYTYPE_SYMBOL(x) ir::BinaryTypeSymbol[static_cast<int>(x)]

/**
 * @brief Type for Logical node
 * @author xupengcheng
 */
enum class LogicalType : int {
#define TYPE_LOGICALTYPE_OP_MAP(opname, op) opname,
#include "x/logical_types.def"
};

static const char *LogicalTypeSymbol[] = {
#define TYPE_LOGICALTYPE_OP_MAP(opname, op) #op,
#include "x/logical_types.def"
};
#define LOGICALTYPE_SYMBOL(x) (::ir::LogicalTypeSymbol[static_cast<int>(x)])

/**
 * @brief Type for Unary node
 * @author xupengcheng
 */
enum class UnaryType : int {
#define TYPE_UNARYTYPE_OP_MAP(opname, op) opname,
#include "x/unary_types.def"
};
static const char *UnaryTypeSymbols[] = {
#define TYPE_UNARYTYPE_OP_MAP(opname, op) #op,
#include "x/unary_types.def"
};
#define UNARYTYPE_SYMBOL(x) ir::UnaryTypeSymbols[static_cast<int>(x)]

/**
 * @brief Type for Scalars.
 * @author xupengcheng
 */
enum class ScalarType : int {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  SCALARTYPE_NAME,
#include "x/scalar_types.def"
};

static const char *ScalarTypeSymbols[]{
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, type) #native_type,
#include "x/scalar_types.def"
};
#define SCALARTYPE_SYMBOL(x) (::ir::ScalarTypeSymbols[static_cast<int>(x)])

static std::unordered_map<int, std::string> unary_convertor{
#define TYPE_STRING_TO_TYPE_UNARY(NAME, CLASS) \
  {static_cast<int>(ir::UnaryType::CLASS), #NAME},
#include "x/tensor_op_types.def"
};
#define UNARY_SYMBOL(x) unary_convertor[static_cast<int>(x)]

static std::unordered_map<int, std::string> binary_convertor{
#define TYPE_STRING_TO_TYPE_BINARY(NAME, CLASS) \
  {static_cast<int>(ir::BinaryType::CLASS), #NAME},
#include "x/tensor_op_types.def"
};
#define BINARY_SYMBOL(x) binary_convertor[static_cast<int>(x)]

static std::unordered_map<int, std::string> logical_convertor{
#define TYPE_STRING_TO_TYPE_LOGICAL(NAME, CLASS) \
  {static_cast<int>(ir::LogicalType::CLASS), #NAME},
#include "x/tensor_op_types.def"
};
#define LOGICAL_SYMBOL(x) logical_convertor[static_cast<int>(x)]

static std::unordered_map<int, std::string> scalartype_convertor{
#define TYPE_STRING_TO_TYPE_DATA(NAME, CLASS) \
  {static_cast<int>(ir::ScalarType::CLASS), #NAME},
#include "x/tensor_op_types.def"
};
#define SCALAR_SYMBOL(x) scalartype_convertor[static_cast<int>(x)]

static std::unordered_map<std::string, ir::ScalarType> scalar_mapping{
#define TYPE_STRING_TO_TYPE_DATA(NAME, CLASS) {#NAME, ir::ScalarType::CLASS},
#include "x/tensor_op_types.def"
};

template <typename T>
struct get_mapped_type {};

#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  template <>                                                       \
  struct get_mapped_type<NATIVE_TYPE> {                             \
    static const ScalarType type = ScalarType::SCALARTYPE_NAME;     \
  };
#include "x/scalar_types.def"

#define CHECK_DATA_TYPE(expr, type)                          \
  ELENA_ASSERT_EQ(expr->get_dtype(), ::ir::ScalarType::type, \
                  "Data type mismatch");
#define CHECK_SAME_DATA_TYPE(A, B) \
  ELENA_ASSERT_EQ((A)->get_dtype(), (B)->get_dtype(), "Data type mismatch");

/**
 * @brief Attr Types
 *
 */
enum class AttrType : int {
#define TYPE_ATTR(type) type,
#include "x/attr_types.def"
};

static const char *AttrTypeNames[] = {
#define TYPE_ATTR(type) #type,
#include "x/attr_types.def"
};
#define ATTRTYPE_NAME(x) (::ir::AttrTypeNames[static_cast<int>(x)])

/**
 * @brief Type attached to stage
 * @author lichuandong
 */
enum class AttachType : int {
  GroupRoot,
  Inline,
  InlinedAlready,
  Scope,
  // ScanUpdate
};
enum class IterAttrType : int {
  Data,
  Vectorized,
  Unrolled,
  Parallelized,
  Thread
};

}  // namespace ir

namespace backend {
enum class TargetType : int {
  x86,
  NVGPU,
  AMDGPU,
  Cambricon,
  Cambricon_RTC,
  NVGPU_RTC,
  AMDGPU_RTC
};

}  // namespace backend

#endif  // ELENA_INCLUDE_IR_TYPE_H_
