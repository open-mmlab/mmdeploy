//===-- elena/include/codegen/DeviceCodegen.h
// - Code generate for device kernel -------*- C++ -*-===//
//
// Part of the Elena Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the DeviceCodegen class, which is used
/// to generate device kernel code.
///
//===----------------------------------------------------------------------===//

#ifndef ELENA_INCLUDE_CODEGEN_DEVICECODEGEN_H_
#define ELENA_INCLUDE_CODEGEN_DEVICECODEGEN_H_
#include <set>
#include <string>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CodeGen/TextGen.h"
#include "IR/VisitorBase.h"
#include "api.h"

using namespace ir;  // NOLINT

// clang-format off
#define UNARY_TYPE_DEVICE(X, Y) \
  X(BitwiseNot, ~)            \
  X(Negate, -)                \
  X(Not, !)                   \
  Y(Abs, abs)                 \
  Y(Sqr, sqr)                 \
  Y(Sqrt, sqrt)               \
  Y(Floor, floor)             \
  Y(Ceil, ceil)               \
  Y(Exp, exp)                 \
  Y(Log, log)                 \
  Y(Log2, log2)               \
  Y(Log10, log10)             \
  Y(Log1p, log1p)             \
  Y(Rsqrt, rsqrt)             \
  Y(IsInf, is_inf)            \
  Y(IsNan, is_nan)            \
  Y(Sinh, sinh)               \
  Y(Cosh, cosh)               \
  Y(Tanh, tanh)               \
  Y(Asin, asin)               \
  Y(Acos, acos)               \
  Y(Atan, atan)               \
  Y(Asinh, asinh)             \
  Y(Acosh, acosh)             \
  Y(Atanh, atanh)             \
  Y(Expm1, expm1)             \
  Y(Round, round)             \
  Y(Cround, cround)           \
  Y(Sin, sin)                 \
  Y(Cos, cos)                 \
  Y(Tan, tan)                 \
  Y(Relu, relu)               \
  Y(Selu, selu)               \
  Y(Sigmoid, sigmoid)         \
  Y(Sign, sign)               \
  Y(Sign2, sign2)
// clang-format on

#define SECOND_AS_STRING(UNARY, OP) #OP,
static const char *UOpNames[] = {
    UNARY_TYPE_DEVICE(SECOND_AS_STRING, SECOND_AS_STRING)};
#define UOP_DEVICE_NAME(x) (UOpNames[static_cast<int>(x)])

#define SECOND_APPEND_F(UNARY, OP) #OP "f",
static const char *UOpNames_f[] = {
    UNARY_TYPE_DEVICE(SECOND_AS_STRING, SECOND_APPEND_F)};
#define UOP_DEVICE_NAME_F(x) (UOpNames_f[static_cast<int>(x)])

#define SECOND_APPEND_H(UNARY, OP) "h" #OP,
static const char *h_UOpNames[] = {
    UNARY_TYPE_DEVICE(SECOND_AS_STRING, SECOND_APPEND_H)};
#define H_UOP_DEVICE_NAME(x) (h_UOpNames[static_cast<int>(x)])

#define TYPE_OF(expr) SCALARTYPE_SYMBOL((expr)->get_dtype())

#define ENABLE_SORTER false

class CudaCode;
class TangCode;
class HipCode;
class CambriconCode;
class CCode;

///
/// \brief Codegen for device kernel code.
/// \tparam DeviceType
template <typename DeviceType>
class DeviceCodegen final : public VisitorBase<DeviceCodegen<DeviceType>>,
                            public TextGen {
 public:
  explicit DeviceCodegen(std::ostream &output_stream,
                         std::set<int> var_mentioned = std::set<int>{});

  explicit DeviceCodegen(std::ostream &output_stream,
                         std::set<std::string> str_var_mentioned =
                             std::set<std::string>{});

  ///
  /// \brief Mark the current tensor var as be visited
  /// \param p
  void markVisited(TensorVar *node) { markVisitedByName(node->get_name()); }

  ///
  /// \brief Mark the current tensor var as be visited by name
  /// \param p
  void markVisitedByName(std::string n) { WellDefined.insert(std::move(n)); }

  template <typename T>
  void visit(ir::Const<T> *node);

  void visit(ir::Allocate *node);
  void visit(ir::Provide *node);
  void visit(ir::For *node);
  void visit(ir::Store *node);
  void visit(ir::Attr *node);
  void visit(ir::TensorVar *node);
  void visit(ir::Binary *node);
  void visit(ir::Unary *node);
  void visit(ir::IterVar *node);
  void visit(ir::ScalarVar *node);
  void visit(ir::Let *node);
  void visit(ir::IfThenElse *node);
  void visit(ir::Logical *node);
  void visit(ir::Evaluate *node);
  void visit(ir::Select *);
  void visit(ir::Call *node);
  void visit(ir::Cast *node);
  void visit(ir::BroadcastSymbol *node);
  void visit(ir::ScalarAssign *node);

  // for raw code gen
  void visit(ComputeOp *);
  void visit(Reduce *);

  using VisitorBase<DeviceCodegen<DeviceType>>::visit;
  int FloatType = 0;   // set half flag   0:Float32/64  1:Float16   2:BFloat16
  int AtomicType = 0;  // set atomic for diff-type  0:int  1:float  2:double
  std::vector<ir::ScalarVarPtr> vars;

  std::set<std::string> WellDefined;

 private:
  std::set<int> VarMentioned;
  std::set<std::string> StrVarMentioned;

  std::unordered_map<std::string, std::string> CudaMemory;

  std::unordered_map<std::string, std::string> CambriconMemory;
};

enum float_type { FLOAT = 0, FP16, BF16 };

// rounds is complicated realization of round(in parrots), while cround is
// realized as normal round in device
const char Prelude[] = R"(
#include "elena_int.h"
)";

const char PreludeTang[] = R"(
#include "tang.h"
)";

static constexpr const char *CudaHalfUtil = R"(
// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// fix undefined fp16 match function
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
static inline __device__  half hpow(half x, half y) {
  float tmp_x = __half2float(x);
  float tmp_y = __half2float(y);
  float result = powf(tmp_x, tmp_y);
  return __float2half(result);
}

static inline __device__  half htanh(half x) {
  float tmp_x = __half2float(x);
  float result = tanhf(tmp_x);
  return __float2half(result);
}
#else
static inline __host__ half hpow(half x, half y);
static inline  __host__ half htanh(half x);
#endif
)";

static constexpr const char *CudaBF16Util = R"(
// Pack two bfloat16 values.
static inline __device__ __host__ unsigned
__pack_nv_bfloat162(const nv_bfloat16 x, const nv_bfloat16 y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// fix undefined fp16 match function
static inline __device__ __host__ nv_bfloat16 hpow(nv_bfloat16 x, nv_bfloat16 y) {
  float tmp_x = __bfloat162float(x);
  float tmp_y = __bfloat162float(y);
  float result = powf(tmp_x, tmp_y);
  return __float2bfloat16(result);
}

static inline __device__ __host__ nv_bfloat16 htanh(nv_bfloat16 x) {
  float tmp_x = __bfloat162float(x);
  float result = tanhf(tmp_x);
  return __float2bfloat16(result);
}
)";

static constexpr const char *CudaHeader = R"(
#define int8_t char
#define int16_t short
#define int32_t int
#define int64_t long int
#define uint8_t unsigned char
#define uint16_t unsigned short
#define uint32_t unsigned int
#define uint64_t unsigned long int
#define sqrf(x) (x * x)
#define signf(x) (x > 0) - ( x < 0 )
)";

static constexpr const char *FloatHeader = R"(
#define floor(x) floorf(x)
#define sign2f(x) (fmax((float)0, (float)signf(x)))
#define reluf(x) fmax(x, 0)
#define seluf(x)                                       \
    (1.0507009873554804934193349852946                \
     * (x > 0 ? x : 1.6732632423543772848170429916717 \
                    * (exp(x) - 1)))
#define sigmoidf(x) (1 / (1 + exp(-x)))
// #define remainder(x, y) (signf(y) + signf(fmod(x, y)) == 0 ? fmod(x, y) + y : fmod(x, y))
#define remainder(x, y) ((x)- (y)*floor((x)/(y)))
#define either_or(x, y, z) ((x==0)? y:z)
#define mul_nan2zero(x, y) (((x==0 && isnan(y)) || (y==0 && isnan(x)))? 0 : x * y)
#define rrelu_bin(x, y) (x>=0? x: x*y)
#define rrelu_rand(x, y) (x + (y-x)*generateRandom(25))
#define reach(x,y) (x>=y?1:0)
#define beyond(x,y) (x>y?1:0)
#define same(x,y) (x==y?1:0)
#define sign_mul(x, y) (x >= 0? fabs(y) : -fabs(y))
#define rounds(x) ((int)(x)&1? roundf(x):   \
(x>=0? (x-floorf(x)>0.5?ceilf(x):floorf(x)): \
(ceilf(x)-x>0.5?floorf(x):ceilf(x))))
#define reverse01(x,y) x>0?1:y
#define ceil_uint64(x) (static_cast<uint64_t>((ceilf(x))))
#define ceilf(x) ((((x) - (static_cast<int>(x))) > 1e-6) ? (floorf(x) + (1)) : (floorf(x)))
)";

static constexpr const char *Fp16Header = R"(
//#define sign2f(x) (max((half)0, (half)signf(x)))
#define reluf(x) max((half)x, (half)0)
)";

static constexpr const char *BF16Header = R"(
//#define sign2f(x) (max((nv_bfloat16)0, (nv_bfloat16)signf(x)))
#define reluf(x) max((nv_bfloat16)x, (nv_bfloat16)0)
)";

static constexpr const char *Common16BitHeader = R"(
#define seluf(x)                                       \
    (1.0507009873554804934193349852946                \
     * (x > 0 ? x : 1.6732632423543772848170429916717 \
                    * (hexp(x) - 1)))
#define sigmoidf(x) (1 / (1 + hexp(-x)))
// #define remainder(x, y) (signf(y) + signf(fmod(x, y)) == 0 ? fmod(x, y) + y : fmod(x, y))
#define remainder(x, y) ((x)- (y)*hfloor((x)/(y)))
#define either_or(x, y, z) ((x==0)? y:z)
#define mul_nan2zero(x, y) (((x==0 && __hisnan(y)) || (y==0 && __hisnan(x)))? 0 : x * y)
#define rrelu_bin(x, y) (x>=0? x: x*y)
//#define rrelu_rand(x, y) (x + (y-x)*generateRandom(25))
#define reach(x,y) (x>=y?1:0)
#define beyond(x,y) (x>y?1:0)
#define same(x,y) (x==y?1:0)
#define rounds(x) ((int)(x)&1? roundf((float)x):   \
(x>=0? (x-hfloor(x)>0.5?hceil(x):hfloor(x)): \
(hceil(x)-x>0.5?hfloor(x):hceil(x))))
#define reverse01(x,y) x>0?1:y
)";

static constexpr const char *TangHeader = R"(
#include <stddef.h>
#define __global__ __Tglobal__
#define __device__ __Tdevice__
#define __constant__ __Tconstant__
#define __host__ __Thost__
#define __shared__ __Tshared__
#define __Tconstant__ __attribute__((Tconstant))
#define __Tdevice__ __attribute__((Tdevice))
#define __Tglobal__ __attribute__((Tglobal))
#define __Thost__ __attribute__((Thost))
#define __Tshared__ __attribute__((Tshared))
#define __forceinline__ __inline__ __attribute__((always_inline))
struct dim3 {
    unsigned x;
    unsigned y;
    unsigned z;
    __Thost__ __Tdevice__ dim3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};
struct uint3 {
    unsigned x;
    unsigned y;
    unsigned z;
    __Thost__ __Tdevice__ uint3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};
//#define __syncthreads()
// #include <vector_types.h>
)";

static constexpr const char *atomicFloat = R"(
__device__ static float atomicMaxFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMinFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

)";

static constexpr const char *atomicDouble = R"(
__device__ double atomicMaxDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMinDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}
)";

#endif  // ELENA_INCLUDE_CODEGEN_DEVICECODEGEN_H_
