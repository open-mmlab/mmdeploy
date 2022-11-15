// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_MARCO_H_
#define MMDEPLOY_SRC_CORE_MARCO_H_

#ifndef MMDEPLOY_EXPORT
#ifdef _MSC_VER
#define MMDEPLOY_EXPORT __declspec(dllexport)
#else
#define MMDEPLOY_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifndef MMDEPLOY_API
#ifdef MMDEPLOY_API_EXPORTS
#define MMDEPLOY_API MMDEPLOY_EXPORT
#else
#define MMDEPLOY_API
#endif
#endif

#define _MMDEPLOY_PP_CONCAT_IMPL(s1, s2) s1##s2
#define MMDEPLOY_PP_CONCAT(s1, s2) _MMDEPLOY_PP_CONCAT_IMPL(s1, s2)

#define MMDEPLOY_PP_EXPAND(...) __VA_ARGS__

// ! Be aware of ODR violation when using __COUNTER__
#ifdef __COUNTER__
#define MMDEPLOY_ANONYMOUS_VARIABLE(str) MMDEPLOY_PP_CONCAT(str, __COUNTER__)
#else
#define MMDEPLOY_ANONYMOUS_VARIABLE(str) MMDEPLOY_PP_CONCAT(str, __LINE__)
#endif

#define MMDEPLOY_PP_NARG(...) _MMDEPLOY_PP_NARG(__VA_ARGS__, _MMDEPLOY_PP_RESQ_N())

#define _MMDEPLOY_PP_NARG(...) _MMDEPLOY_PP_ARG_N(__VA_ARGS__)

#define _MMDEPLOY_PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, \
                           _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30,  \
                           _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44,  \
                           _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58,  \
                           _59, _60, _61, _62, _63, N, ...)                                       \
  N

#define _MMDEPLOY_PP_RESQ_N()                                                                     \
  63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, \
      39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, \
      16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define MMDEPLOY_PP_MAP_1(f, x) f(x)
#define MMDEPLOY_PP_MAP_2(f, x, ...) f(x), MMDEPLOY_PP_MAP_1(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_3(f, x, ...) f(x), MMDEPLOY_PP_MAP_2(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_4(f, x, ...) f(x), MMDEPLOY_PP_MAP_3(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_5(f, x, ...) f(x), MMDEPLOY_PP_MAP_4(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_6(f, x, ...) f(x), MMDEPLOY_PP_MAP_5(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_7(f, x, ...) f(x), MMDEPLOY_PP_MAP_6(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_8(f, x, ...) f(x), MMDEPLOY_PP_MAP_7(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_9(f, x, ...) f(x), MMDEPLOY_PP_MAP_8(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_10(f, x, ...) f(x), MMDEPLOY_PP_MAP_9(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_11(f, x, ...) f(x), MMDEPLOY_PP_MAP_10(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_12(f, x, ...) f(x), MMDEPLOY_PP_MAP_11(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_13(f, x, ...) f(x), MMDEPLOY_PP_MAP_12(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_14(f, x, ...) f(x), MMDEPLOY_PP_MAP_13(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_15(f, x, ...) f(x), MMDEPLOY_PP_MAP_14(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_16(f, x, ...) f(x), MMDEPLOY_PP_MAP_15(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_17(f, x, ...) f(x), MMDEPLOY_PP_MAP_16(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_18(f, x, ...) f(x), MMDEPLOY_PP_MAP_17(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_19(f, x, ...) f(x), MMDEPLOY_PP_MAP_18(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_20(f, x, ...) f(x), MMDEPLOY_PP_MAP_19(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_21(f, x, ...) f(x), MMDEPLOY_PP_MAP_20(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_22(f, x, ...) f(x), MMDEPLOY_PP_MAP_21(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_23(f, x, ...) f(x), MMDEPLOY_PP_MAP_22(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_24(f, x, ...) f(x), MMDEPLOY_PP_MAP_23(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_25(f, x, ...) f(x), MMDEPLOY_PP_MAP_24(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_26(f, x, ...) f(x), MMDEPLOY_PP_MAP_25(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_27(f, x, ...) f(x), MMDEPLOY_PP_MAP_26(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_28(f, x, ...) f(x), MMDEPLOY_PP_MAP_27(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_29(f, x, ...) f(x), MMDEPLOY_PP_MAP_28(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_30(f, x, ...) f(x), MMDEPLOY_PP_MAP_29(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_31(f, x, ...) f(x), MMDEPLOY_PP_MAP_30(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_32(f, x, ...) f(x), MMDEPLOY_PP_MAP_31(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_33(f, x, ...) f(x), MMDEPLOY_PP_MAP_32(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_34(f, x, ...) f(x), MMDEPLOY_PP_MAP_33(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_35(f, x, ...) f(x), MMDEPLOY_PP_MAP_34(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_36(f, x, ...) f(x), MMDEPLOY_PP_MAP_35(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_37(f, x, ...) f(x), MMDEPLOY_PP_MAP_36(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_38(f, x, ...) f(x), MMDEPLOY_PP_MAP_37(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_39(f, x, ...) f(x), MMDEPLOY_PP_MAP_38(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_40(f, x, ...) f(x), MMDEPLOY_PP_MAP_39(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_41(f, x, ...) f(x), MMDEPLOY_PP_MAP_40(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_42(f, x, ...) f(x), MMDEPLOY_PP_MAP_41(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_43(f, x, ...) f(x), MMDEPLOY_PP_MAP_42(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_44(f, x, ...) f(x), MMDEPLOY_PP_MAP_43(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_45(f, x, ...) f(x), MMDEPLOY_PP_MAP_44(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_46(f, x, ...) f(x), MMDEPLOY_PP_MAP_45(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_47(f, x, ...) f(x), MMDEPLOY_PP_MAP_46(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_48(f, x, ...) f(x), MMDEPLOY_PP_MAP_47(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_49(f, x, ...) f(x), MMDEPLOY_PP_MAP_48(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_50(f, x, ...) f(x), MMDEPLOY_PP_MAP_49(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_51(f, x, ...) f(x), MMDEPLOY_PP_MAP_50(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_52(f, x, ...) f(x), MMDEPLOY_PP_MAP_51(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_53(f, x, ...) f(x), MMDEPLOY_PP_MAP_52(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_54(f, x, ...) f(x), MMDEPLOY_PP_MAP_53(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_55(f, x, ...) f(x), MMDEPLOY_PP_MAP_54(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_56(f, x, ...) f(x), MMDEPLOY_PP_MAP_55(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_57(f, x, ...) f(x), MMDEPLOY_PP_MAP_56(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_58(f, x, ...) f(x), MMDEPLOY_PP_MAP_57(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_59(f, x, ...) f(x), MMDEPLOY_PP_MAP_58(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_60(f, x, ...) f(x), MMDEPLOY_PP_MAP_59(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_61(f, x, ...) f(x), MMDEPLOY_PP_MAP_60(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_62(f, x, ...) f(x), MMDEPLOY_PP_MAP_61(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_63(f, x, ...) f(x), MMDEPLOY_PP_MAP_62(f, __VA_ARGS__)
#define MMDEPLOY_PP_MAP_64(f, x, ...) f(x), MMDEPLOY_PP_MAP_63(f, __VA_ARGS__)

#define MMDEPLOY_PP_MAP(f, ...) \
  _MMDEPLOY_PP_MAP_IMPL1(f, MMDEPLOY_PP_NARG(__VA_ARGS__), __VA_ARGS__)

#define _MMDEPLOY_PP_MAP_IMPL1(f, n, ...) \
  _MMDEPLOY_PP_MAP_IMPL2(f, MMDEPLOY_PP_CONCAT(MMDEPLOY_PP_MAP_, n), __VA_ARGS__)

#define _MMDEPLOY_PP_MAP_IMPL2(f, M_, ...) M_(f, __VA_ARGS__)

#endif  // MMDEPLOY_SRC_CORE_MARCO_H_
