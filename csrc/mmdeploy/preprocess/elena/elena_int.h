#include <math.h>

#define min(X, Y) ((X) < (Y) ? (X) : (Y))
#define max(X, Y) ((X) > (Y) ? (X) : (Y))

#define int8_t char
#define int16_t short
#define int32_t int
#define int64_t long int
#define uint8_t unsigned char
#define uint16_t unsigned short
#define uint32_t unsigned int
#define uint64_t unsigned long int
#define sqrf(x) (x * x)
#define signf(x) (x > 0) - (x < 0)

#define floor(x) floorf(x)
#define sign2f(x) (fmax((float)0, (float)signf(x)))
#define reluf(x) fmax(x, 0)
#define seluf(x)                       \
  (1.0507009873554804934193349852946 * \
   (x > 0 ? x : 1.6732632423543772848170429916717 * (exp(x) - 1)))
#define sigmoidf(x) (1 / (1 + exp(-x)))
// #define remainder(x, y) (signf(y) + signf(fmod(x, y)) == 0 ? fmod(x, y) + y : fmod(x, y))
#define remainder(x, y) ((x) - (y)*floor((x) / (y)))
#define either_or(x, y, z) ((x == 0) ? y : z)
#define mul_nan2zero(x, y) (((x == 0 && isnan(y)) || (y == 0 && isnan(x))) ? 0 : x * y)
#define rrelu_bin(x, y) (x >= 0 ? x : x * y)
#define rrelu_rand(x, y) (x + (y - x) * generateRandom(25))
#define reach(x, y) (x >= y ? 1 : 0)
#define beyond(x, y) (x > y ? 1 : 0)
#define same(x, y) (x == y ? 1 : 0)
#define sign_mul(x, y) (x >= 0 ? fabs(y) : -fabs(y))
#define rounds(x)                                                      \
  ((int)(x)&1 ? roundf(x)                                              \
              : (x >= 0 ? (x - floorf(x) > 0.5 ? ceilf(x) : floorf(x)) \
                        : (ceilf(x) - x > 0.5 ? floorf(x) : ceilf(x))))
#define reverse01(x, y) x > 0 ? 1 : y
#define ceil_uint64(x) (static_cast<uint64_t>((ceilf(x))))
#define ceilf(x) ((((x) - (static_cast<int>(x))) > 1e-6) ? (floorf(x) + (1)) : (floorf(x)))
