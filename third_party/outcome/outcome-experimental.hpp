/* A less simple result type
(C) 2018-2019 Niall Douglas <http://www.nedproductions.biz/> (17 commits)
File Created: Apr 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_EXPERIMENTAL_STATUS_OUTCOME_HPP
#define OUTCOME_EXPERIMENTAL_STATUS_OUTCOME_HPP
/* A less simple result type
(C) 2017-2020 Niall Douglas <http://www.nedproductions.biz/> (20 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_OUTCOME_HPP
#define OUTCOME_BASIC_OUTCOME_HPP
/* Configure Outcome with QuickCppLib
(C) 2015-2021 Niall Douglas <http://www.nedproductions.biz/> (24 commits)
File Created: August 2015


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_V2_CONFIG_HPP
#define OUTCOME_V2_CONFIG_HPP
/* Sets Outcome version
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (4 commits)


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
/*! AWAITING HUGO JSON CONVERSION TOOL */
#define OUTCOME_VERSION_MAJOR 2
/*! AWAITING HUGO JSON CONVERSION TOOL */
#define OUTCOME_VERSION_MINOR 2
/*! AWAITING HUGO JSON CONVERSION TOOL */
#define OUTCOME_VERSION_PATCH 0
/*! AWAITING HUGO JSON CONVERSION TOOL */
#define OUTCOME_VERSION_REVISION 0 // Revision version for cmake and DLL version stamping
/*! AWAITING HUGO JSON CONVERSION TOOL */
#ifndef OUTCOME_DISABLE_ABI_PERMUTATION
#define OUTCOME_UNSTABLE_VERSION
#endif
// Pull in detection of __MINGW64_VERSION_MAJOR
#if defined(__MINGW32__) && !0L
#include <_mingw.h>
#endif
/* Configure QuickCppLib
(C) 2016-2017 Niall Douglas <http://www.nedproductions.biz/> (8 commits)


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef QUICKCPPLIB_CONFIG_HPP
#define QUICKCPPLIB_CONFIG_HPP
/* Provides SG-10 feature checking for all C++ compilers
(C) 2014-2017 Niall Douglas <http://www.nedproductions.biz/> (13 commits)
File Created: Nov 2014


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef QUICKCPPLIB_HAS_FEATURE_H
#define QUICKCPPLIB_HAS_FEATURE_H
#if __cplusplus >= 201103L
// Some of these macros ended up getting removed by ISO standards,
// they are prefixed with ////
////#if !defined(__cpp_alignas)
////#define __cpp_alignas 190000
////#endif
////#if !defined(__cpp_default_function_template_args)
////#define __cpp_default_function_template_args 190000
////#endif
////#if !defined(__cpp_defaulted_functions)
////#define __cpp_defaulted_functions 190000
////#endif
////#if !defined(__cpp_deleted_functions)
////#define __cpp_deleted_functions 190000
////#endif
////#if !defined(__cpp_generalized_initializers)
////#define __cpp_generalized_initializers 190000
////#endif
////#if !defined(__cpp_implicit_moves)
////#define __cpp_implicit_moves 190000
////#endif
////#if !defined(__cpp_inline_namespaces)
////#define __cpp_inline_namespaces 190000
////#endif
////#if !defined(__cpp_local_type_template_args)
////#define __cpp_local_type_template_args 190000
////#endif
////#if !defined(__cpp_noexcept)
////#define __cpp_noexcept 190000
////#endif
////#if !defined(__cpp_nonstatic_member_init)
////#define __cpp_nonstatic_member_init 190000
////#endif
////#if !defined(__cpp_nullptr)
////#define __cpp_nullptr 190000
////#endif
////#if !defined(__cpp_override_control)
////#define __cpp_override_control 190000
////#endif
////#if !defined(__cpp_thread_local)
////#define __cpp_thread_local 190000
////#endif
////#if !defined(__cpp_auto_type)
////#define __cpp_auto_type 190000
////#endif
////#if !defined(__cpp_strong_enums)
////#define __cpp_strong_enums 190000
////#endif
////#if !defined(__cpp_trailing_return)
////#define __cpp_trailing_return 190000
////#endif
////#if !defined(__cpp_unrestricted_unions)
////#define __cpp_unrestricted_unions 190000
////#endif
#if !defined(__cpp_alias_templates)
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr)
#if __cplusplus >= 201402L
#define __cpp_constexpr 201304 // relaxed constexpr
#else
#define __cpp_constexpr 190000
#endif
#endif
#if !defined(__cpp_decltype)
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors)
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) //// renamed from __cpp_explicit_conversions
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors)
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) //// NEW
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas)
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi)
#define __cpp_nsdmi 190000 //// NEW
#endif
#if !defined(__cpp_range_based_for) //// renamed from __cpp_range_for
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings)
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) //// renamed from __cpp_reference_qualified_functions
#define __cpp_ref_qualifiers 190000
#endif
#if !defined(__cpp_rvalue_references)
#define __cpp_rvalue_references 190000
#endif
#if !defined(__cpp_static_assert)
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_unicode_characters) //// NEW
#define __cpp_unicode_characters 190000
#endif
#if !defined(__cpp_unicode_literals)
#define __cpp_unicode_literals 190000
#endif
#if !defined(__cpp_user_defined_literals)
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates)
#define __cpp_variadic_templates 190000
#endif
#endif
#if __cplusplus >= 201402L
// Some of these macros ended up getting removed by ISO standards,
// they are prefixed with ////
////#if !defined(__cpp_contextual_conversions)
////#define __cpp_contextual_conversions 190000
////#endif
////#if !defined(__cpp_digit_separators)
////#define __cpp_digit_separators 190000
////#endif
////#if !defined(__cpp_relaxed_constexpr)
////#define __cpp_relaxed_constexpr 190000
////#endif
////#if !defined(__cpp_runtime_arrays)
////# define __cpp_runtime_arrays 190000
////#endif
#if !defined(__cpp_aggregate_nsdmi)
#define __cpp_aggregate_nsdmi 190000
#endif
#if !defined(__cpp_binary_literals)
#define __cpp_binary_literals 190000
#endif
#if !defined(__cpp_decltype_auto)
#define __cpp_decltype_auto 190000
#endif
#if !defined(__cpp_generic_lambdas)
#define __cpp_generic_lambdas 190000
#endif
#if !defined(__cpp_init_captures)
#define __cpp_init_captures 190000
#endif
#if !defined(__cpp_return_type_deduction)
#define __cpp_return_type_deduction 190000
#endif
#if !defined(__cpp_sized_deallocation)
#define __cpp_sized_deallocation 190000
#endif
#if !defined(__cpp_variable_templates)
#define __cpp_variable_templates 190000
#endif
#endif
// VS2010: _MSC_VER=1600
// VS2012: _MSC_VER=1700
// VS2013: _MSC_VER=1800
// VS2015: _MSC_VER=1900
// VS2017: _MSC_VER=1910
#if defined(_MSC_VER) && !defined(__clang__)
#if !defined(__cpp_exceptions) && defined(_CPPUNWIND)
#define __cpp_exceptions 190000
#endif
#if !defined(__cpp_rtti) && defined(_CPPRTTI)
#define __cpp_rtti 190000
#endif
// C++ 11
#if !defined(__cpp_alias_templates) && _MSC_VER >= 1800
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr) && _MSC_FULL_VER >= 190023506 /* VS2015 */
#define __cpp_constexpr 190000
#endif
#if !defined(__cpp_decltype) && _MSC_VER >= 1600
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors) && _MSC_VER >= 1800
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) && _MSC_VER >= 1800
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors) && _MSC_VER >= 1900
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) && _MSC_VER >= 1900
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas) && _MSC_VER >= 1600
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi) && _MSC_VER >= 1900
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for) && _MSC_VER >= 1700
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings) && _MSC_VER >= 1800
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) && _MSC_VER >= 1900
#define __cpp_ref_qualifiers 190000
#endif
#if !defined(__cpp_rvalue_references) && _MSC_VER >= 1600
#define __cpp_rvalue_references 190000
#endif
#if !defined(__cpp_static_assert) && _MSC_VER >= 1600
#define __cpp_static_assert 190000
#endif
//#if !defined(__cpp_unicode_literals)
//# define __cpp_unicode_literals 190000
//#endif
#if !defined(__cpp_user_defined_literals) && _MSC_VER >= 1900
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates) && _MSC_VER >= 1800
#define __cpp_variadic_templates 190000
#endif
// C++ 14
//#if !defined(__cpp_aggregate_nsdmi)
//#define __cpp_aggregate_nsdmi 190000
//#endif
#if !defined(__cpp_binary_literals) && _MSC_VER >= 1900
#define __cpp_binary_literals 190000
#endif
#if !defined(__cpp_decltype_auto) && _MSC_VER >= 1900
#define __cpp_decltype_auto 190000
#endif
#if !defined(__cpp_generic_lambdas) && _MSC_VER >= 1900
#define __cpp_generic_lambdas 190000
#endif
#if !defined(__cpp_init_captures) && _MSC_VER >= 1900
#define __cpp_init_captures 190000
#endif
#if !defined(__cpp_return_type_deduction) && _MSC_VER >= 1900
#define __cpp_return_type_deduction 190000
#endif
#if !defined(__cpp_sized_deallocation) && _MSC_VER >= 1900
#define __cpp_sized_deallocation 190000
#endif
#if !defined(__cpp_variable_templates) && _MSC_FULL_VER >= 190023506
#define __cpp_variable_templates 190000
#endif
#endif // _MSC_VER
// Much to my surprise, GCC's support of these is actually incomplete, so fill in the gaps
#if (defined(__GNUC__) && !defined(__clang__))
#define QUICKCPPLIB_GCC (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if !defined(__cpp_exceptions) && defined(__EXCEPTIONS)
#define __cpp_exceptions 190000
#endif
#if !defined(__cpp_rtti) && defined(__GXX_RTTI)
#define __cpp_rtti 190000
#endif
// C++ 11
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
#if !defined(__cpp_alias_templates) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes) && (QUICKCPPLIB_GCC >= 40800)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr) && (QUICKCPPLIB_GCC >= 40600)
#define __cpp_constexpr 190000
#endif
#if !defined(__cpp_decltype) && (QUICKCPPLIB_GCC >= 40300)
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors) && (QUICKCPPLIB_GCC >= 40800)
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) && (QUICKCPPLIB_GCC >= 40800)
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for) && (QUICKCPPLIB_GCC >= 40600)
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) && (QUICKCPPLIB_GCC >= 40801)
#define __cpp_ref_qualifiers 190000
#endif
// __cpp_rvalue_reference deviation
#if !defined(__cpp_rvalue_references) && defined(__cpp_rvalue_reference)
#define __cpp_rvalue_references __cpp_rvalue_reference
#endif
#if !defined(__cpp_static_assert) && (QUICKCPPLIB_GCC >= 40300)
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_unicode_characters) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_unicode_characters 190000
#endif
#if !defined(__cpp_unicode_literals) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_unicode_literals 190000
#endif
#if !defined(__cpp_user_defined_literals) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates) && (QUICKCPPLIB_GCC >= 40400)
#define __cpp_variadic_templates 190000
#endif
// C++ 14
// Every C++ 14 supporting GCC does the right thing here
#endif // __GXX_EXPERIMENTAL_CXX0X__
#endif // GCC
// clang deviates in some places from the present SG-10 draft, plus older
// clangs are quite incomplete
#if defined(__clang__)
#define QUICKCPPLIB_CLANG (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#if !defined(__cpp_exceptions) && (defined(__EXCEPTIONS) || defined(_CPPUNWIND))
#define __cpp_exceptions 190000
#endif
#if !defined(__cpp_rtti) && (defined(__GXX_RTTI) || defined(_CPPRTTI))
#define __cpp_rtti 190000
#endif
// C++ 11
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
#if !defined(__cpp_alias_templates) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes) && (QUICKCPPLIB_CLANG >= 30300)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_constexpr 190000
#endif
#if !defined(__cpp_decltype) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors) && (QUICKCPPLIB_CLANG >= 30300)
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_range_based_for 190000
#endif
// __cpp_raw_string_literals deviation
#if !defined(__cpp_raw_strings) && defined(__cpp_raw_string_literals)
#define __cpp_raw_strings __cpp_raw_string_literals
#endif
#if !defined(__cpp_raw_strings) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_ref_qualifiers 190000
#endif
// __cpp_rvalue_reference deviation
#if !defined(__cpp_rvalue_references) && defined(__cpp_rvalue_reference)
#define __cpp_rvalue_references __cpp_rvalue_reference
#endif
#if !defined(__cpp_rvalue_references) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_rvalue_references 190000
#endif
#if !defined(__cpp_static_assert) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_unicode_characters) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_unicode_characters 190000
#endif
#if !defined(__cpp_unicode_literals) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_unicode_literals 190000
#endif
// __cpp_user_literals deviation
#if !defined(__cpp_user_defined_literals) && defined(__cpp_user_literals)
#define __cpp_user_defined_literals __cpp_user_literals
#endif
#if !defined(__cpp_user_defined_literals) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_variadic_templates 190000
#endif
// C++ 14
// Every C++ 14 supporting clang does the right thing here
#endif // __GXX_EXPERIMENTAL_CXX0X__
#endif // clang
#endif
#ifndef QUICKCPPLIB_DISABLE_ABI_PERMUTATION
// Note the second line of this file must ALWAYS be the git SHA, third line ALWAYS the git SHA update time
#define QUICKCPPLIB_PREVIOUS_COMMIT_REF 9a151aab3abaf1ced6b0b82f9328beb536386254
#define QUICKCPPLIB_PREVIOUS_COMMIT_DATE "2021-02-23 11:25:30 +00:00"
#define QUICKCPPLIB_PREVIOUS_COMMIT_UNIQUE 9a151aab
#endif
#define QUICKCPPLIB_VERSION_GLUE2(a, b) a##b
#define QUICKCPPLIB_VERSION_GLUE(a, b) QUICKCPPLIB_VERSION_GLUE2(a, b)
// clang-format off
#if defined(QUICKCPPLIB_DISABLE_ABI_PERMUTATION)
#define QUICKCPPLIB_NAMESPACE quickcpplib
#define QUICKCPPLIB_NAMESPACE_BEGIN namespace quickcpplib {
#define QUICKCPPLIB_NAMESPACE_END }
#else
#define QUICKCPPLIB_NAMESPACE quickcpplib::QUICKCPPLIB_VERSION_GLUE(_, QUICKCPPLIB_PREVIOUS_COMMIT_UNIQUE)
#define QUICKCPPLIB_NAMESPACE_BEGIN namespace quickcpplib { namespace QUICKCPPLIB_VERSION_GLUE(_, QUICKCPPLIB_PREVIOUS_COMMIT_UNIQUE) {
#define QUICKCPPLIB_NAMESPACE_END } }
#endif
// clang-format on
#ifdef _MSC_VER
#define QUICKCPPLIB_BIND_MESSAGE_PRAGMA2(x) __pragma(message(x))
#define QUICKCPPLIB_BIND_MESSAGE_PRAGMA(x) QUICKCPPLIB_BIND_MESSAGE_PRAGMA2(x)
#define QUICKCPPLIB_BIND_MESSAGE_PREFIX(type) __FILE__ "(" QUICKCPPLIB_BIND_STRINGIZE2(__LINE__) "): " type ": "
#define QUICKCPPLIB_BIND_MESSAGE_(type, prefix, msg) QUICKCPPLIB_BIND_MESSAGE_PRAGMA(prefix msg)
#else
#define QUICKCPPLIB_BIND_MESSAGE_PRAGMA2(x) _Pragma(#x)
#define QUICKCPPLIB_BIND_MESSAGE_PRAGMA(type, x) QUICKCPPLIB_BIND_MESSAGE_PRAGMA2(type x)
#define QUICKCPPLIB_BIND_MESSAGE_(type, prefix, msg) QUICKCPPLIB_BIND_MESSAGE_PRAGMA(type, msg)
#endif
//! Have the compiler output a message
#define QUICKCPPLIB_MESSAGE(msg) QUICKCPPLIB_BIND_MESSAGE_(message, QUICKCPPLIB_BIND_MESSAGE_PREFIX("message"), msg)
//! Have the compiler output a note
#define QUICKCPPLIB_NOTE(msg) QUICKCPPLIB_BIND_MESSAGE_(message, QUICKCPPLIB_BIND_MESSAGE_PREFIX("note"), msg)
//! Have the compiler output a warning
#define QUICKCPPLIB_WARNING(msg) QUICKCPPLIB_BIND_MESSAGE_(GCC warning, QUICKCPPLIB_BIND_MESSAGE_PREFIX("warning"), msg)
//! Have the compiler output an error
#define QUICKCPPLIB_ERROR(msg) QUICKCPPLIB_BIND_MESSAGE_(GCC error, QUICKCPPLIB_BIND_MESSAGE_PREFIX("error"), msg)
#define QUICKCPPLIB_ANNOTATE_RWLOCK_CREATE(p)
#define QUICKCPPLIB_ANNOTATE_RWLOCK_DESTROY(p)
#define QUICKCPPLIB_ANNOTATE_RWLOCK_ACQUIRED(p, s)
#define QUICKCPPLIB_ANNOTATE_RWLOCK_RELEASED(p, s)
#define QUICKCPPLIB_ANNOTATE_IGNORE_READS_BEGIN()
#define QUICKCPPLIB_ANNOTATE_IGNORE_READS_END()
#define QUICKCPPLIB_ANNOTATE_IGNORE_WRITES_BEGIN()
#define QUICKCPPLIB_ANNOTATE_IGNORE_WRITES_END()
#define QUICKCPPLIB_DRD_IGNORE_VAR(x)
#define QUICKCPPLIB_DRD_STOP_IGNORING_VAR(x)
#define QUICKCPPLIB_RUNNING_ON_VALGRIND (0)
#ifndef QUICKCPPLIB_IN_THREAD_SANITIZER
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define QUICKCPPLIB_IN_THREAD_SANITIZER 1
#endif
#elif defined(__SANITIZE_THREAD__)
#define QUICKCPPLIB_IN_THREAD_SANITIZER 1
#endif
#endif
#ifndef QUICKCPPLIB_IN_THREAD_SANITIZER
#define QUICKCPPLIB_IN_THREAD_SANITIZER 0
#endif
#if QUICKCPPLIB_IN_THREAD_SANITIZER
#define QUICKCPPLIB_DISABLE_THREAD_SANITIZE __attribute__((no_sanitize_thread))
#else
#define QUICKCPPLIB_DISABLE_THREAD_SANITIZE
#endif
#ifndef QUICKCPPLIB_SMT_PAUSE
#if !defined(__clang__) && defined(_MSC_VER) && _MSC_VER >= 1310 && (defined(_M_IX86) || defined(_M_X64))
extern "C" void _mm_pause();
#pragma intrinsic(_mm_pause)
#define QUICKCPPLIB_SMT_PAUSE _mm_pause();
#elif !defined(__c2__) && defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
#define QUICKCPPLIB_SMT_PAUSE __asm__ __volatile__("rep; nop" : : : "memory");
#endif
#endif
#ifndef QUICKCPPLIB_FORCEINLINE
#if defined(_MSC_VER)
#define QUICKCPPLIB_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#define QUICKCPPLIB_FORCEINLINE __attribute__((always_inline))
#else
#define QUICKCPPLIB_FORCEINLINE
#endif
#endif
#ifndef QUICKCPPLIB_NOINLINE
#if defined(_MSC_VER)
#define QUICKCPPLIB_NOINLINE __declspec(noinline)
#elif defined(__GNUC__)
#define QUICKCPPLIB_NOINLINE __attribute__((noinline))
#else
#define QUICKCPPLIB_NOINLINE
#endif
#endif
#ifdef __has_cpp_attribute
#define QUICKCPPLIB_HAS_CPP_ATTRIBUTE(attr) __has_cpp_attribute(attr)
#else
#define QUICKCPPLIB_HAS_CPP_ATTRIBUTE(attr) (0)
#endif
#if !defined(QUICKCPPLIB_NORETURN)
#if QUICKCPPLIB_HAS_CPP_ATTRIBUTE(noreturn)
#define QUICKCPPLIB_NORETURN [[noreturn]]
#elif defined(_MSC_VER)
#define QUICKCPPLIB_NORETURN __declspec(noreturn)
#elif defined(__GNUC__)
#define QUICKCPPLIB_NORETURN __attribute__((__noreturn__))
#else
#define QUICKCPPLIB_NORETURN
#endif
#endif
#ifndef QUICKCPPLIB_NODISCARD
#if 0L || (_HAS_CXX17 && _MSC_VER >= 1911 /* VS2017.3 */)
#define QUICKCPPLIB_NODISCARD [[nodiscard]]
#endif
#endif
#ifndef QUICKCPPLIB_NODISCARD
#if QUICKCPPLIB_HAS_CPP_ATTRIBUTE(nodiscard)
#define QUICKCPPLIB_NODISCARD [[nodiscard]]
#elif defined(__clang__) // deliberately not GCC
#define QUICKCPPLIB_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER)
// _Must_inspect_result_ expands into this
#define QUICKCPPLIB_NODISCARD __declspec("SAL_name" "(" "\"_Must_inspect_result_\"" "," "\"\"" "," "\"2\"" ")") __declspec("SAL_begin") __declspec("SAL_post") __declspec("SAL_mustInspect") __declspec("SAL_post") __declspec("SAL_checkReturn") __declspec("SAL_end")
#endif
#endif
#ifndef QUICKCPPLIB_NODISCARD
#define QUICKCPPLIB_NODISCARD
#endif
#ifndef QUICKCPPLIB_SYMBOL_VISIBLE
#if defined(_MSC_VER)
#define QUICKCPPLIB_SYMBOL_VISIBLE
#elif defined(__GNUC__)
#define QUICKCPPLIB_SYMBOL_VISIBLE __attribute__((visibility("default")))
#else
#define QUICKCPPLIB_SYMBOL_VISIBLE
#endif
#endif
#ifndef QUICKCPPLIB_SYMBOL_EXPORT
#if defined(_MSC_VER)
#define QUICKCPPLIB_SYMBOL_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define QUICKCPPLIB_SYMBOL_EXPORT __attribute__((visibility("default")))
#else
#define QUICKCPPLIB_SYMBOL_EXPORT
#endif
#endif
#ifndef QUICKCPPLIB_SYMBOL_IMPORT
#if defined(_MSC_VER)
#define QUICKCPPLIB_SYMBOL_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#define QUICKCPPLIB_SYMBOL_IMPORT
#else
#define QUICKCPPLIB_SYMBOL_IMPORT
#endif
#endif
#ifndef QUICKCPPLIB_THREAD_LOCAL
#if _MSC_VER >= 1800
#define QUICKCPPLIB_THREAD_LOCAL_IS_CXX11 1
#elif __cplusplus >= 201103L
#if __GNUC__ >= 5 && !defined(__clang__)
#define QUICKCPPLIB_THREAD_LOCAL_IS_CXX11 1
#elif defined(__has_feature)
#if __has_feature(cxx_thread_local)
#define QUICKCPPLIB_THREAD_LOCAL_IS_CXX11 1
#endif
#endif
#endif
#ifdef QUICKCPPLIB_THREAD_LOCAL_IS_CXX11
#define QUICKCPPLIB_THREAD_LOCAL thread_local
#endif
#ifndef QUICKCPPLIB_THREAD_LOCAL
#if defined(_MSC_VER)
#define QUICKCPPLIB_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__)
#define QUICKCPPLIB_THREAD_LOCAL __thread
#else
#error Unknown compiler, cannot set QUICKCPPLIB_THREAD_LOCAL
#endif
#endif
#endif
/* MSVC capable preprocessor macro overloading
(C) 2014-2017 Niall Douglas <http://www.nedproductions.biz/> (3 commits)
File Created: Aug 2014


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef QUICKCPPLIB_PREPROCESSOR_MACRO_OVERLOAD_H
#define QUICKCPPLIB_PREPROCESSOR_MACRO_OVERLOAD_H
#define QUICKCPPLIB_GLUE(x, y) x y
#define QUICKCPPLIB_RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, count, ...) count
#define QUICKCPPLIB_EXPAND_ARGS(args) QUICKCPPLIB_RETURN_ARG_COUNT args
#define QUICKCPPLIB_COUNT_ARGS_MAX8(...) QUICKCPPLIB_EXPAND_ARGS((__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define QUICKCPPLIB_OVERLOAD_MACRO2(name, count) name##count
#define QUICKCPPLIB_OVERLOAD_MACRO1(name, count) QUICKCPPLIB_OVERLOAD_MACRO2(name, count)
#define QUICKCPPLIB_OVERLOAD_MACRO(name, count) QUICKCPPLIB_OVERLOAD_MACRO1(name, count)
#define QUICKCPPLIB_CALL_OVERLOAD(name, ...) QUICKCPPLIB_GLUE(QUICKCPPLIB_OVERLOAD_MACRO(name, QUICKCPPLIB_COUNT_ARGS_MAX8(__VA_ARGS__)), (__VA_ARGS__))
#define QUICKCPPLIB_GLUE_(x, y) x y
#define QUICKCPPLIB_RETURN_ARG_COUNT_(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, count, ...) count
#define QUICKCPPLIB_EXPAND_ARGS_(args) QUICKCPPLIB_RETURN_ARG_COUNT_ args
#define QUICKCPPLIB_COUNT_ARGS_MAX8_(...) QUICKCPPLIB_EXPAND_ARGS_((__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define QUICKCPPLIB_OVERLOAD_MACRO2_(name, count) name##count
#define QUICKCPPLIB_OVERLOAD_MACRO1_(name, count) QUICKCPPLIB_OVERLOAD_MACRO2_(name, count)
#define QUICKCPPLIB_OVERLOAD_MACRO_(name, count) QUICKCPPLIB_OVERLOAD_MACRO1_(name, count)
#define QUICKCPPLIB_CALL_OVERLOAD_(name, ...) QUICKCPPLIB_GLUE_(QUICKCPPLIB_OVERLOAD_MACRO_(name, QUICKCPPLIB_COUNT_ARGS_MAX8_(__VA_ARGS__)), (__VA_ARGS__))
#endif
#if defined(__cpp_concepts) && !defined(QUICKCPPLIB_DISABLE_CONCEPTS_SUPPORT)
#define QUICKCPPLIB_TREQUIRES_EXPAND8(a, b, c, d, e, f, g, h) a &&QUICKCPPLIB_TREQUIRES_EXPAND7(b, c, d, e, f, g, h)
#define QUICKCPPLIB_TREQUIRES_EXPAND7(a, b, c, d, e, f, g) a &&QUICKCPPLIB_TREQUIRES_EXPAND6(b, c, d, e, f, g)
#define QUICKCPPLIB_TREQUIRES_EXPAND6(a, b, c, d, e, f) a &&QUICKCPPLIB_TREQUIRES_EXPAND5(b, c, d, e, f)
#define QUICKCPPLIB_TREQUIRES_EXPAND5(a, b, c, d, e) a &&QUICKCPPLIB_TREQUIRES_EXPAND4(b, c, d, e)
#define QUICKCPPLIB_TREQUIRES_EXPAND4(a, b, c, d) a &&QUICKCPPLIB_TREQUIRES_EXPAND3(b, c, d)
#define QUICKCPPLIB_TREQUIRES_EXPAND3(a, b, c) a &&QUICKCPPLIB_TREQUIRES_EXPAND2(b, c)
#define QUICKCPPLIB_TREQUIRES_EXPAND2(a, b) a &&QUICKCPPLIB_TREQUIRES_EXPAND1(b)
#define QUICKCPPLIB_TREQUIRES_EXPAND1(a) a
//! Expands into a && b && c && ...
#define QUICKCPPLIB_TREQUIRES(...) requires QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_TREQUIRES_EXPAND, __VA_ARGS__)
#define QUICKCPPLIB_TEMPLATE(...) template <__VA_ARGS__>
#define QUICKCPPLIB_TEXPR(...) requires { (__VA_ARGS__); }
#define QUICKCPPLIB_TPRED(...) (__VA_ARGS__)
#if !defined(_MSC_VER) || _MSC_FULL_VER >= 192400000 // VS 2019 16.3 is broken here
#define QUICKCPPLIB_REQUIRES(...) requires(__VA_ARGS__)
#else
#define QUICKCPPLIB_REQUIRES(...)
#endif
#else
#define QUICKCPPLIB_TEMPLATE(...) template <__VA_ARGS__
#define QUICKCPPLIB_TREQUIRES(...) , __VA_ARGS__ >
#define QUICKCPPLIB_TEXPR(...) typename = decltype(__VA_ARGS__)
#ifdef _MSC_VER
// MSVC gives an error if every specialisation of a template is always ill-formed, so
// the more powerful SFINAE form below causes pukeage :(
#define QUICKCPPLIB_TPRED(...) typename = typename std::enable_if<(__VA_ARGS__)>::type
#else
#define QUICKCPPLIB_TPRED(...) typename std::enable_if<(__VA_ARGS__), bool>::type = true
#endif
#define QUICKCPPLIB_REQUIRES(...)
#endif
#endif
#ifndef __cpp_variadic_templates
#error Outcome needs variadic template support in the compiler
#endif
#if __cpp_constexpr < 201304 && _MSC_FULL_VER < 191100000
#error Outcome needs constexpr (C++ 14) support in the compiler
#endif
#ifndef __cpp_variable_templates
#error Outcome needs variable template support in the compiler
#endif
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ < 6
#error Due to a bug in nested template variables parsing, Outcome does not work on GCCs earlier than v6.
#endif
#ifndef OUTCOME_SYMBOL_VISIBLE
#define OUTCOME_SYMBOL_VISIBLE QUICKCPPLIB_SYMBOL_VISIBLE
#endif
#ifndef OUTCOME_FORCEINLINE
#define OUTCOME_FORCEINLINE QUICKCPPLIB_FORCEINLINE
#endif
#ifndef OUTCOME_NODISCARD
#define OUTCOME_NODISCARD QUICKCPPLIB_NODISCARD
#endif
#ifndef OUTCOME_THREAD_LOCAL
#define OUTCOME_THREAD_LOCAL QUICKCPPLIB_THREAD_LOCAL
#endif
#ifndef OUTCOME_TEMPLATE
#define OUTCOME_TEMPLATE(...) QUICKCPPLIB_TEMPLATE(__VA_ARGS__)
#endif
#ifndef OUTCOME_TREQUIRES
#define OUTCOME_TREQUIRES(...) QUICKCPPLIB_TREQUIRES(__VA_ARGS__)
#endif
#ifndef OUTCOME_TEXPR
#define OUTCOME_TEXPR(...) QUICKCPPLIB_TEXPR(__VA_ARGS__)
#endif
#ifndef OUTCOME_TPRED
#define OUTCOME_TPRED(...) QUICKCPPLIB_TPRED(__VA_ARGS__)
#endif
#ifndef OUTCOME_REQUIRES
#define OUTCOME_REQUIRES(...) QUICKCPPLIB_REQUIRES(__VA_ARGS__)
#endif
/* Convenience macros for importing local namespace binds
(C) 2014-2017 Niall Douglas <http://www.nedproductions.biz/> (9 commits)
File Created: Aug 2014


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef QUICKCPPLIB_BIND_IMPORT_HPP
#define QUICKCPPLIB_BIND_IMPORT_HPP
/* 2014-10-9 ned: I lost today figuring out the below. I really hate the C preprocessor now.
 *
 * Anyway, infinity = 8. It's easy to expand below if needed.
 */
#define QUICKCPPLIB_BIND_STRINGIZE(a) #a
#define QUICKCPPLIB_BIND_STRINGIZE2(a) QUICKCPPLIB_BIND_STRINGIZE(a)
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION8(a, b, c, d, e, f, g, h) a##_##b##_##c##_##d##_##e##_##f##_##g##_##h
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION7(a, b, c, d, e, f, g) a##_##b##_##c##_##d##_##e##_##f##_##g
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION6(a, b, c, d, e, f) a##_##b##_##c##_##d##_##e##_##f
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION5(a, b, c, d, e) a##_##b##_##c##_##d##_##e
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION4(a, b, c, d) a##_##b##_##c##_##d
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION3(a, b, c) a##_##b##_##c
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION2(a, b) a##_##b
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION1(a) a
//! Concatenates each parameter with _
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_VERSION, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT_2(name, modifier) name
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT2(name, modifier) ::name
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT_1(name) name
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT1(name) ::name
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT_(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_SELECT_, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_SELECT, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND8(a, b, c, d, e, f, g, h) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d QUICKCPPLIB_BIND_NAMESPACE_SELECT e QUICKCPPLIB_BIND_NAMESPACE_SELECT f QUICKCPPLIB_BIND_NAMESPACE_SELECT g QUICKCPPLIB_BIND_NAMESPACE_SELECT h
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND7(a, b, c, d, e, f, g) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d QUICKCPPLIB_BIND_NAMESPACE_SELECT e QUICKCPPLIB_BIND_NAMESPACE_SELECT f QUICKCPPLIB_BIND_NAMESPACE_SELECT g
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND6(a, b, c, d, e, f) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d QUICKCPPLIB_BIND_NAMESPACE_SELECT e QUICKCPPLIB_BIND_NAMESPACE_SELECT f
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND5(a, b, c, d, e) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d QUICKCPPLIB_BIND_NAMESPACE_SELECT e
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND4(a, b, c, d) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND3(a, b, c) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND2(a, b) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND1(a) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a
//! Expands into a::b::c:: ...
#define QUICKCPPLIB_BIND_NAMESPACE(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_EXPAND, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT2(name, modifier) modifier namespace name {
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT1(name) namespace name {
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND8(a, b, c, d, e, f, g, h) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND7(b, c, d, e, f, g, h)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND7(a, b, c, d, e, f, g) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND6(b, c, d, e, f, g)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND6(a, b, c, d, e, f) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND5(b, c, d, e, f)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND5(a, b, c, d, e) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND4(b, c, d, e)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND4(a, b, c, d) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND3(b, c, d)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND3(a, b, c) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND2(b, c)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND2(a, b) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND1(b)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND1(a) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a
//! Expands into namespace a { namespace b { namespace c ...
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT2(name, modifier) modifier namespace name {
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT1(name) export namespace name {
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND8(a, b, c, d, e, f, g, h) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND7(b, c, d, e, f, g, h)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND7(a, b, c, d, e, f, g) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND6(b, c, d, e, f, g)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND6(a, b, c, d, e, f) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND5(b, c, d, e, f)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND5(a, b, c, d, e) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND4(b, c, d, e)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND4(a, b, c, d) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND3(b, c, d)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND3(a, b, c) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND2(b, c)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND2(a, b) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND1(b)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND1(a) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a
//! Expands into export namespace a { namespace b { namespace c ...
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT2(name, modifier) }
#define QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT1(name) }
#define QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND8(a, b, c, d, e, f, g, h) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND7(b, c, d, e, f, g, h)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND7(a, b, c, d, e, f, g) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND6(b, c, d, e, f, g)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND6(a, b, c, d, e, f) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND5(b, c, d, e, f)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND5(a, b, c, d, e) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND4(b, c, d, e)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND4(a, b, c, d) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND3(b, c, d)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND3(a, b, c) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND2(b, c)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND2(a, b) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND1(b)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND1(a) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a
//! Expands into } } ...
#define QUICKCPPLIB_BIND_NAMESPACE_END(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND, __VA_ARGS__)
//! Expands into a static const char string array used to mark BindLib compatible namespaces
#define QUICKCPPLIB_BIND_DECLARE(decl, desc) static const char *quickcpplib_out[] = {#decl, desc};
#endif
#ifndef OUTCOME_ENABLE_LEGACY_SUPPORT_FOR
#define OUTCOME_ENABLE_LEGACY_SUPPORT_FOR 220 // the v2.2 Outcome release
#endif
#if defined(OUTCOME_UNSTABLE_VERSION)
/* UPDATED BY SCRIPT
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (225 commits)


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
// Note the second line of this file must ALWAYS be the git SHA, third line ALWAYS the git SHA update time
#define OUTCOME_PREVIOUS_COMMIT_REF 35644f5cdeb90f6f29d80714a371098d311cbd47
#define OUTCOME_PREVIOUS_COMMIT_DATE "2021-02-26 10:38:25 +00:00"
#define OUTCOME_PREVIOUS_COMMIT_UNIQUE 35644f5c
#define OUTCOME_V2 (QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2, OUTCOME_PREVIOUS_COMMIT_UNIQUE))
#ifdef _DEBUG
#define OUTCOME_V2_CXX_MODULE_NAME QUICKCPPLIB_BIND_NAMESPACE((QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2d, OUTCOME_PREVIOUS_COMMIT_UNIQUE)))
#else
#define OUTCOME_V2_CXX_MODULE_NAME QUICKCPPLIB_BIND_NAMESPACE((QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2, OUTCOME_PREVIOUS_COMMIT_UNIQUE)))
#endif
#else
#define OUTCOME_V2 (QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2))
#ifdef _DEBUG
#define OUTCOME_V2_CXX_MODULE_NAME QUICKCPPLIB_BIND_NAMESPACE((QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2d)))
#else
#define OUTCOME_V2_CXX_MODULE_NAME QUICKCPPLIB_BIND_NAMESPACE((QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2)))
#endif
#endif
#if defined(GENERATING_OUTCOME_MODULE_INTERFACE)
#define OUTCOME_V2_NAMESPACE QUICKCPPLIB_BIND_NAMESPACE(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_BEGIN QUICKCPPLIB_BIND_NAMESPACE_BEGIN(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_EXPORT_BEGIN QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_END QUICKCPPLIB_BIND_NAMESPACE_END(OUTCOME_V2)
#else
#define OUTCOME_V2_NAMESPACE QUICKCPPLIB_BIND_NAMESPACE(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_BEGIN QUICKCPPLIB_BIND_NAMESPACE_BEGIN(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_EXPORT_BEGIN QUICKCPPLIB_BIND_NAMESPACE_BEGIN(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_END QUICKCPPLIB_BIND_NAMESPACE_END(OUTCOME_V2)
#endif
#include <cstdint> // for uint32_t etc
#include <initializer_list>
#include <iosfwd> // for future serialisation
#include <new> // for placement in moves etc
#include <type_traits>
#ifndef OUTCOME_USE_STD_IN_PLACE_TYPE
#if defined(_MSC_VER) && _HAS_CXX17
#define OUTCOME_USE_STD_IN_PLACE_TYPE 1 // MSVC always has std::in_place_type
#elif __cplusplus >= 201700
// libstdc++ before GCC 6 doesn't have it, despite claiming C++ 17 support
#ifdef __has_include
#if !__has_include(<variant>)
#define OUTCOME_USE_STD_IN_PLACE_TYPE 0 // must have it if <variant> is present
#endif
#endif
#ifndef OUTCOME_USE_STD_IN_PLACE_TYPE
#define OUTCOME_USE_STD_IN_PLACE_TYPE 1
#endif
#else
#define OUTCOME_USE_STD_IN_PLACE_TYPE 0
#endif
#endif
#if OUTCOME_USE_STD_IN_PLACE_TYPE
#include <utility> // for in_place_type_t
OUTCOME_V2_NAMESPACE_BEGIN
template <class T> using in_place_type_t = std::in_place_type_t<T>;
using std::in_place_type;
OUTCOME_V2_NAMESPACE_END
#else
OUTCOME_V2_NAMESPACE_BEGIN
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class T> in_place_type_t. Potential doc page: `in_place_type_t<T>`
*/
template <class T> struct in_place_type_t
{
  explicit in_place_type_t() = default;
};
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> constexpr in_place_type_t<T> in_place_type{};
OUTCOME_V2_NAMESPACE_END
#endif
#ifndef OUTCOME_TRIVIAL_ABI
#if 0L || __clang_major__ >= 7
//! Defined to be `[[clang::trivial_abi]]` when on a new enough clang compiler. Usually automatic, can be overriden.
#define OUTCOME_TRIVIAL_ABI [[clang::trivial_abi]]
#else
#define OUTCOME_TRIVIAL_ABI
#endif
#endif
OUTCOME_V2_NAMESPACE_BEGIN
namespace detail
{
  // Test if type is an in_place_type_t
  template <class T> struct is_in_place_type_t
  {
    static constexpr bool value = false;
  };
  template <class U> struct is_in_place_type_t<in_place_type_t<U>>
  {
    static constexpr bool value = true;
  };
  // Replace void with constructible void_type
  struct empty_type
  {
  };
  struct void_type
  {
    // We always compare true to another instance of me
    constexpr bool operator==(void_type /*unused*/) const noexcept { return true; }
    constexpr bool operator!=(void_type /*unused*/) const noexcept { return false; }
  };
  template <class T> using devoid = std::conditional_t<std::is_void<T>::value, void_type, T>;
  template <class Output, class Input> using rebind_type5 = Output;
  template <class Output, class Input>
  using rebind_type4 = std::conditional_t< //
  std::is_volatile<Input>::value, //
  std::add_volatile_t<rebind_type5<Output, std::remove_volatile_t<Input>>>, //
  rebind_type5<Output, Input>>;
  template <class Output, class Input>
  using rebind_type3 = std::conditional_t< //
  std::is_const<Input>::value, //
  std::add_const_t<rebind_type4<Output, std::remove_const_t<Input>>>, //
  rebind_type4<Output, Input>>;
  template <class Output, class Input>
  using rebind_type2 = std::conditional_t< //
  std::is_lvalue_reference<Input>::value, //
  std::add_lvalue_reference_t<rebind_type3<Output, std::remove_reference_t<Input>>>, //
  rebind_type3<Output, Input>>;
  template <class Output, class Input>
  using rebind_type = std::conditional_t< //
  std::is_rvalue_reference<Input>::value, //
  std::add_rvalue_reference_t<rebind_type2<Output, std::remove_reference_t<Input>>>, //
  rebind_type2<Output, Input>>;
  // static_assert(std::is_same_v<rebind_type<int, volatile const double &&>, volatile const int &&>, "");
  /* True if type is the same or constructible. Works around a bug where clang + libstdc++
  pukes on std::is_constructible<filesystem::path, void> (this bug is fixed upstream).
  */
  template <class T, class U> struct _is_explicitly_constructible
  {
    static constexpr bool value = std::is_constructible<T, U>::value;
  };
  template <class T> struct _is_explicitly_constructible<T, void>
  {
    static constexpr bool value = false;
  };
  template <> struct _is_explicitly_constructible<void, void>
  {
    static constexpr bool value = false;
  };
  template <class T, class U> static constexpr bool is_explicitly_constructible = _is_explicitly_constructible<T, U>::value;
  template <class T, class U> struct _is_implicitly_constructible
  {
    static constexpr bool value = std::is_convertible<U, T>::value;
  };
  template <class T> struct _is_implicitly_constructible<T, void>
  {
    static constexpr bool value = false;
  };
  template <> struct _is_implicitly_constructible<void, void>
  {
    static constexpr bool value = false;
  };
  template <class T, class U> static constexpr bool is_implicitly_constructible = _is_implicitly_constructible<T, U>::value;
  template <class T, class... Args> struct _is_nothrow_constructible
  {
    static constexpr bool value = std::is_nothrow_constructible<T, Args...>::value;
  };
  template <class T> struct _is_nothrow_constructible<T, void>
  {
    static constexpr bool value = false;
  };
  template <> struct _is_nothrow_constructible<void, void>
  {
    static constexpr bool value = false;
  };
  template <class T, class... Args> static constexpr bool is_nothrow_constructible = _is_nothrow_constructible<T, Args...>::value;
  template <class T, class... Args> struct _is_constructible
  {
    static constexpr bool value = std::is_constructible<T, Args...>::value;
  };
  template <class T> struct _is_constructible<T, void>
  {
    static constexpr bool value = false;
  };
  template <> struct _is_constructible<void, void>
  {
    static constexpr bool value = false;
  };
  template <class T, class... Args> static constexpr bool is_constructible = _is_constructible<T, Args...>::value;
#ifndef OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE
#if defined(_MSC_VER) && _HAS_CXX17
#define OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE 1 // MSVC always has std::is_nothrow_swappable
#elif __cplusplus >= 201700
// libstdc++ before GCC 6 doesn't have it, despite claiming C++ 17 support
#ifdef __has_include
#if !__has_include(<variant>)
#define OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE 0
#endif
#endif
#ifndef OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE
#define OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE 1
#endif
#else
#define OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE 0
#endif
#endif
// True if type is nothrow swappable
#if !0L && OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE
  template <class T> using is_nothrow_swappable = std::is_nothrow_swappable<T>;
#else
  template <class T> struct is_nothrow_swappable
  {
    static constexpr bool value = std::is_nothrow_move_constructible<T>::value && std::is_nothrow_move_assignable<T>::value;
  };
#endif
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#ifndef OUTCOME_THROW_EXCEPTION
#ifdef __cpp_exceptions
#define OUTCOME_THROW_EXCEPTION(expr) throw expr
#else
#ifdef __ANDROID__
#define OUTCOME_DISABLE_EXECINFO
#endif
#ifndef OUTCOME_DISABLE_EXECINFO
#ifdef _WIN32
/* Implements backtrace() et al from glibc on win64
(C) 2016-2017 Niall Douglas <http://www.nedproductions.biz/> (4 commits)
File Created: Mar 2016


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef BOOST_BINDLIB_EXECINFO_WIN64_H
#define BOOST_BINDLIB_EXECINFO_WIN64_H
#ifndef _WIN32
#error Can only be included on Windows
#endif
#include <sal.h>
#include <stddef.h>
#ifdef QUICKCPPLIB_EXPORTS
#define EXECINFO_DECL extern __declspec(dllexport)
#else
#if defined(__cplusplus) && (!defined(QUICKCPPLIB_HEADERS_ONLY) || QUICKCPPLIB_HEADERS_ONLY == 1) && !0L
#define EXECINFO_DECL inline
#elif defined(QUICKCPPLIB_DYN_LINK) && !defined(QUICKCPPLIB_STATIC_LINK)
#define EXECINFO_DECL extern __declspec(dllimport)
#else
#define EXECINFO_DECL extern
#endif
#endif
#ifdef __cplusplus
extern "C" {
#endif
//! Fill the array of void * at bt with up to len entries, returning entries filled.
EXECINFO_DECL _Check_return_ size_t backtrace(_Out_writes_(len) void **bt, _In_ size_t len);
//! Returns a malloced block of string representations of the input backtrace.
EXECINFO_DECL _Check_return_ _Ret_writes_maybenull_(len) char **backtrace_symbols(_In_reads_(len) void *const *bt, _In_ size_t len);
// extern void backtrace_symbols_fd(void *const *bt, size_t len, int fd);
#ifdef __cplusplus
}
#if (!defined(QUICKCPPLIB_HEADERS_ONLY) || QUICKCPPLIB_HEADERS_ONLY == 1) && !0L
#define QUICKCPPLIB_INCLUDED_BY_HEADER 1
/* Implements backtrace() et al from glibc on win64
(C) 2016-2017 Niall Douglas <http://www.nedproductions.biz/> (14 commits)
File Created: Mar 2016


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
/* Implements backtrace() et al from glibc on win64
(C) 2016-2017 Niall Douglas <http://www.nedproductions.biz/> (4 commits)
File Created: Mar 2016


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#include <atomic>
#include <stdlib.h> // for abort
#include <string.h>
// To avoid including windows.h, this source has been macro expanded and win32 function shimmed for C++ only
#if defined(__cplusplus) && !defined(__clang__)
namespace win32
{
  extern _Ret_maybenull_ void *__stdcall LoadLibraryA(_In_ const char *lpLibFileName);
  typedef int(__stdcall *GetProcAddress_returntype)();
  extern GetProcAddress_returntype __stdcall GetProcAddress(_In_ void *hModule, _In_ const char *lpProcName);
  extern _Success_(return != 0) unsigned short __stdcall RtlCaptureStackBackTrace(_In_ unsigned long FramesToSkip, _In_ unsigned long FramesToCapture,
                                                                                  _Out_writes_to_(FramesToCapture, return ) void **BackTrace,
                                                                                  _Out_opt_ unsigned long *BackTraceHash);
  extern _Success_(return != 0)
  _When_((cchWideChar == -1) && (cbMultiByte != 0),
         _Post_equal_to_(_String_length_(lpMultiByteStr) +
                         1)) int __stdcall WideCharToMultiByte(_In_ unsigned int CodePage, _In_ unsigned long dwFlags, const wchar_t *lpWideCharStr,
                                                               _In_ int cchWideChar, _Out_writes_bytes_to_opt_(cbMultiByte, return ) char *lpMultiByteStr,
                                                               _In_ int cbMultiByte, _In_opt_ const char *lpDefaultChar, _Out_opt_ int *lpUsedDefaultChar);
#pragma comment(lib, "kernel32.lib")
#if (defined(__x86_64__) || defined(_M_X64)) || (defined(__aarch64__) || defined(_M_ARM64))
#pragma comment(linker, "/alternatename:?LoadLibraryA@win32@@YAPEAXPEBD@Z=LoadLibraryA")
#pragma comment(linker, "/alternatename:?GetProcAddress@win32@@YAP6AHXZPEAXPEBD@Z=GetProcAddress")
#pragma comment(linker, "/alternatename:?RtlCaptureStackBackTrace@win32@@YAGKKPEAPEAXPEAK@Z=RtlCaptureStackBackTrace")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@@YAHIKPEB_WHPEADHPEBDPEAH@Z=WideCharToMultiByte")
#elif defined(__x86__) || defined(_M_IX86) || defined(__i386__)
#pragma comment(linker, "/alternatename:?LoadLibraryA@win32@@YGPAXPBD@Z=__imp__LoadLibraryA@4")
#pragma comment(linker, "/alternatename:?GetProcAddress@win32@@YGP6GHXZPAXPBD@Z=__imp__GetProcAddress@8")
#pragma comment(linker, "/alternatename:?RtlCaptureStackBackTrace@win32@@YGGKKPAPAXPAK@Z=__imp__RtlCaptureStackBackTrace@16")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@@YGHIKPB_WHPADHPBDPAH@Z=__imp__WideCharToMultiByte@32")
#elif defined(__arm__) || defined(_M_ARM)
#pragma comment(linker, "/alternatename:?LoadLibraryA@win32@@YAPAXPBD@Z=LoadLibraryA")
#pragma comment(linker, "/alternatename:?GetProcAddress@win32@@YAP6AHXZPAXPBD@Z=GetProcAddress")
#pragma comment(linker, "/alternatename:?RtlCaptureStackBackTrace@win32@@YAGKKPAPAXPAK@Z=RtlCaptureStackBackTrace")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@@YAHIKPB_WHPADHPBDPAH@Z=WideCharToMultiByte")
#else
#error Unknown architecture
#endif
} // namespace win32
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
#ifdef __cplusplus
namespace
{
#endif
  typedef struct _IMAGEHLP_LINE64
  {
    unsigned long SizeOfStruct;
    void *Key;
    unsigned long LineNumber;
    wchar_t *FileName;
    unsigned long long int Address;
  } IMAGEHLP_LINE64, *PIMAGEHLP_LINE64;
  typedef int(__stdcall *SymInitialize_t)(_In_ void *hProcess, _In_opt_ const wchar_t *UserSearchPath, _In_ int fInvadeProcess);
  typedef int(__stdcall *SymGetLineFromAddr64_t)(_In_ void *hProcess, _In_ unsigned long long int dwAddr, _Out_ unsigned long *pdwDisplacement,
                                                 _Out_ PIMAGEHLP_LINE64 Line);
  static std::atomic<unsigned> dbghelp_init_lock;
#if defined(__cplusplus) && !defined(__clang__)
  static void *dbghelp;
#else
static HMODULE dbghelp;
#endif
  static SymInitialize_t SymInitialize;
  static SymGetLineFromAddr64_t SymGetLineFromAddr64;
  static void load_dbghelp()
  {
#if defined(__cplusplus) && !defined(__clang__)
    using win32::GetProcAddress;
    using win32::LoadLibraryA;
#endif
    while(dbghelp_init_lock.exchange(1, std::memory_order_acq_rel))
      ;
    if(dbghelp)
    {
      dbghelp_init_lock.store(0, std::memory_order_release);
      return;
    }
    dbghelp = LoadLibraryA("DBGHELP.DLL");
    if(dbghelp)
    {
      SymInitialize = (SymInitialize_t) GetProcAddress(dbghelp, "SymInitializeW");
      if(!SymInitialize)
        abort();
      if(!SymInitialize((void *) (size_t) -1 /*GetCurrentProcess()*/, NULL, 1))
        abort();
      SymGetLineFromAddr64 = (SymGetLineFromAddr64_t) GetProcAddress(dbghelp, "SymGetLineFromAddrW64");
      if(!SymGetLineFromAddr64)
        abort();
    }
    dbghelp_init_lock.store(0, std::memory_order_release);
  }
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C"
{
#endif
  _Check_return_ size_t backtrace(_Out_writes_(len) void **bt, _In_ size_t len)
  {
#if defined(__cplusplus) && !defined(__clang__)
    using win32::RtlCaptureStackBackTrace;
#endif
    return RtlCaptureStackBackTrace(1, (unsigned long) len, bt, NULL);
  }
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6385 6386) // MSVC static analyser can't grok this function. clang's analyser gives it thumbs up.
#endif
  _Check_return_ _Ret_writes_maybenull_(len) char **backtrace_symbols(_In_reads_(len) void *const *bt, _In_ size_t len)
  {
#if defined(__cplusplus) && !defined(__clang__)
    using win32::WideCharToMultiByte;
#endif
    size_t bytes = (len + 1) * sizeof(void *) + 256, n;
    if(!len)
      return NULL;
    else
    {
      char **ret = (char **) malloc(bytes);
      char *p = (char *) (ret + len + 1), *end = (char *) ret + bytes;
      if(!ret)
        return NULL;
      for(n = 0; n < len + 1; n++)
        ret[n] = NULL;
      load_dbghelp();
      for(n = 0; n < len; n++)
      {
        unsigned long displ;
        IMAGEHLP_LINE64 ihl;
        memset(&ihl, 0, sizeof(ihl));
        ihl.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        int please_realloc = 0;
        if(!bt[n])
        {
          ret[n] = NULL;
        }
        else
        {
          // Keep offset till later
          ret[n] = (char *) ((char *) p - (char *) ret);
          if(!SymGetLineFromAddr64 || !SymGetLineFromAddr64((void *) (size_t) -1 /*GetCurrentProcess()*/, (size_t) bt[n], &displ, &ihl))
          {
            if(n == 0)
            {
              free(ret);
              return NULL;
            }
            ihl.FileName = (wchar_t *) L"unknown";
            ihl.LineNumber = 0;
          }
        retry:
          if(please_realloc)
          {
            char **temp = (char **) realloc(ret, bytes + 256);
            if(!temp)
            {
              free(ret);
              return NULL;
            }
            p = (char *) temp + (p - (char *) ret);
            ret = temp;
            bytes += 256;
            end = (char *) ret + bytes;
          }
          if(ihl.FileName && ihl.FileName[0])
          {
            int plen = WideCharToMultiByte(65001 /*CP_UTF8*/, 0, ihl.FileName, -1, p, (int) (end - p), NULL, NULL);
            if(!plen)
            {
              please_realloc = 1;
              goto retry;
            }
            p[plen - 1] = 0;
            p += plen - 1;
          }
          else
          {
            if(end - p < 16)
            {
              please_realloc = 1;
              goto retry;
            }
            _ui64toa_s((size_t) bt[n], p, end - p, 16);
            p = strchr(p, 0);
          }
          if(end - p < 16)
          {
            please_realloc = 1;
            goto retry;
          }
          *p++ = ':';
          _itoa_s(ihl.LineNumber, p, end - p, 10);
          p = strchr(p, 0) + 1;
        }
      }
      for(n = 0; n < len; n++)
      {
        if(ret[n])
          ret[n] = (char *) ret + (size_t) ret[n];
      }
      return ret;
    }
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  // extern void backtrace_symbols_fd(void *const *bt, size_t len, int fd);
#ifdef __cplusplus
}
#endif
#undef QUICKCPPLIB_INCLUDED_BY_HEADER
#endif
#endif
#endif
#else
#include <execinfo.h>
#endif
#endif // OUTCOME_DISABLE_EXECINFO
#include <cstdio>
#include <cstdlib>
OUTCOME_V2_NAMESPACE_BEGIN
namespace detail
{
  QUICKCPPLIB_NORETURN inline void do_fatal_exit(const char *expr)
  {
#if !defined(OUTCOME_DISABLE_EXECINFO)
    void *bt[16];
    size_t btlen = backtrace(bt, sizeof(bt) / sizeof(bt[0])); // NOLINT
#endif
    fprintf(stderr, "FATAL: Outcome throws exception %s with exceptions disabled\n", expr); // NOLINT
#if !defined(OUTCOME_DISABLE_EXECINFO)
    char **bts = backtrace_symbols(bt, btlen); // NOLINT
    if(bts != nullptr)
    {
      for(size_t n = 0; n < btlen; n++)
      {
        fprintf(stderr, "  %s\n", bts[n]); // NOLINT
      }
      free(bts); // NOLINT
    }
#endif
    abort();
  }
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#define OUTCOME_THROW_EXCEPTION(expr) OUTCOME_V2_NAMESPACE::detail::do_fatal_exit(#expr), (void) (expr)
#endif
#endif
#ifndef BOOST_OUTCOME_AUTO_TEST_CASE
#define BOOST_OUTCOME_AUTO_TEST_CASE(a, b) BOOST_AUTO_TEST_CASE(a, b)
#endif
#endif
/* A very simple result type
(C) 2017-2021 Niall Douglas <http://www.nedproductions.biz/> (14 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_RESULT_HPP
#define OUTCOME_BASIC_RESULT_HPP
/* Says how to convert value, error and exception types
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (12 commits)
File Created: Nov 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_CONVERT_HPP
#define OUTCOME_CONVERT_HPP
/* Storage for a very simple basic_result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (6 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_RESULT_STORAGE_HPP
#define OUTCOME_BASIC_RESULT_STORAGE_HPP
/* Type sugar for success and failure
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (25 commits)
File Created: July 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_SUCCESS_FAILURE_HPP
#define OUTCOME_SUCCESS_FAILURE_HPP
OUTCOME_V2_NAMESPACE_BEGIN
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class T> success_type. Potential doc page: `success_type<T>`
*/
template <class T> struct OUTCOME_NODISCARD success_type
{
  using value_type = T;
private:
  value_type _value;
  uint16_t _spare_storage{0};
public:
  success_type() = default;
  success_type(const success_type &) = default;
  success_type(success_type &&) = default; // NOLINT
  success_type &operator=(const success_type &) = default;
  success_type &operator=(success_type &&) = default; // NOLINT
  ~success_type() = default;
  OUTCOME_TEMPLATE(class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_same<success_type, std::decay_t<U>>::value))
  constexpr explicit success_type(U &&v, uint16_t spare_storage = 0)
      : _value(static_cast<U &&>(v)) // NOLINT
      , _spare_storage(spare_storage)
  {
  }
  constexpr value_type &value() & { return _value; }
  constexpr const value_type &value() const & { return _value; }
  constexpr value_type &&value() && { return static_cast<value_type &&>(_value); }
  constexpr const value_type &&value() const && { return static_cast<value_type &&>(_value); }
  constexpr uint16_t spare_storage() const { return _spare_storage; }
};
template <> struct OUTCOME_NODISCARD success_type<void>
{
  using value_type = void;
  constexpr uint16_t spare_storage() const { return 0; }
};
/*! Returns type sugar for implicitly constructing a `basic_result<T>` with a successful state,
default constructing `T` if necessary.
*/
inline constexpr success_type<void> success() noexcept
{
  return success_type<void>{};
}
/*! Returns type sugar for implicitly constructing a `basic_result<T>` with a successful state.
\effects Copies or moves the successful state supplied into the returned type sugar.
*/
template <class T> inline constexpr success_type<std::decay_t<T>> success(T &&v, uint16_t spare_storage = 0)
{
  return success_type<std::decay_t<T>>{static_cast<T &&>(v), spare_storage};
}
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class EC, class E = void> failure_type. Potential doc page: `failure_type<EC, EP = void>`
*/
template <class EC, class E = void> struct OUTCOME_NODISCARD failure_type
{
  using error_type = EC;
  using exception_type = E;
private:
  error_type _error;
  exception_type _exception;
  bool _have_error{false}, _have_exception{false};
  uint16_t _spare_storage{0};
  struct error_init_tag
  {
  };
  struct exception_init_tag
  {
  };
public:
  failure_type() = default;
  failure_type(const failure_type &) = default;
  failure_type(failure_type &&) = default; // NOLINT
  failure_type &operator=(const failure_type &) = default;
  failure_type &operator=(failure_type &&) = default; // NOLINT
  ~failure_type() = default;
  template <class U, class V>
  constexpr explicit failure_type(U &&u, V &&v, uint16_t spare_storage = 0)
      : _error(static_cast<U &&>(u))
      , _exception(static_cast<V &&>(v))
      , _have_error(true)
      , _have_exception(true)
      , _spare_storage(spare_storage)
  {
  }
  template <class U>
  constexpr explicit failure_type(in_place_type_t<error_type> /*unused*/, U &&u, uint16_t spare_storage = 0, error_init_tag /*unused*/ = error_init_tag())
      : _error(static_cast<U &&>(u))
      , _exception()
      , _have_error(true)
      , _spare_storage(spare_storage)
  {
  }
  template <class U>
  constexpr explicit failure_type(in_place_type_t<exception_type> /*unused*/, U &&u, uint16_t spare_storage = 0,
                                  exception_init_tag /*unused*/ = exception_init_tag())
      : _error()
      , _exception(static_cast<U &&>(u))
      , _have_exception(true)
      , _spare_storage(spare_storage)
  {
  }
  constexpr bool has_error() const { return _have_error; }
  constexpr bool has_exception() const { return _have_exception; }
  constexpr error_type &error() & { return _error; }
  constexpr const error_type &error() const & { return _error; }
  constexpr error_type &&error() && { return static_cast<error_type &&>(_error); }
  constexpr const error_type &&error() const && { return static_cast<error_type &&>(_error); }
  constexpr exception_type &exception() & { return _exception; }
  constexpr const exception_type &exception() const & { return _exception; }
  constexpr exception_type &&exception() && { return static_cast<exception_type &&>(_exception); }
  constexpr const exception_type &&exception() const && { return static_cast<exception_type &&>(_exception); }
  constexpr uint16_t spare_storage() const { return _spare_storage; }
};
template <class EC> struct OUTCOME_NODISCARD failure_type<EC, void>
{
  using error_type = EC;
  using exception_type = void;
private:
  error_type _error;
  uint16_t _spare_storage{0};
public:
  failure_type() = default;
  failure_type(const failure_type &) = default;
  failure_type(failure_type &&) = default; // NOLINT
  failure_type &operator=(const failure_type &) = default;
  failure_type &operator=(failure_type &&) = default; // NOLINT
  ~failure_type() = default;
  OUTCOME_TEMPLATE(class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_same<failure_type, std::decay_t<U>>::value))
  constexpr explicit failure_type(U &&u, uint16_t spare_storage = 0)
      : _error(static_cast<U &&>(u)) // NOLINT
      , _spare_storage(spare_storage)
  {
  }
  constexpr error_type &error() & { return _error; }
  constexpr const error_type &error() const & { return _error; }
  constexpr error_type &&error() && { return static_cast<error_type &&>(_error); }
  constexpr const error_type &&error() const && { return static_cast<error_type &&>(_error); }
  constexpr uint16_t spare_storage() const { return _spare_storage; }
};
template <class E> struct OUTCOME_NODISCARD failure_type<void, E>
{
  using error_type = void;
  using exception_type = E;
private:
  exception_type _exception;
  uint16_t _spare_storage{0};
public:
  failure_type() = default;
  failure_type(const failure_type &) = default;
  failure_type(failure_type &&) = default; // NOLINT
  failure_type &operator=(const failure_type &) = default;
  failure_type &operator=(failure_type &&) = default; // NOLINT
  ~failure_type() = default;
  OUTCOME_TEMPLATE(class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_same<failure_type, std::decay_t<V>>::value))
  constexpr explicit failure_type(V &&v, uint16_t spare_storage = 0)
      : _exception(static_cast<V &&>(v)) // NOLINT
      , _spare_storage(spare_storage)
  {
  }
  constexpr exception_type &exception() & { return _exception; }
  constexpr const exception_type &exception() const & { return _exception; }
  constexpr exception_type &&exception() && { return static_cast<exception_type &&>(_exception); }
  constexpr const exception_type &&exception() const && { return static_cast<exception_type &&>(_exception); }
  constexpr uint16_t spare_storage() const { return _spare_storage; }
};
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class EC> inline constexpr failure_type<std::decay_t<EC>> failure(EC &&v, uint16_t spare_storage = 0)
{
  return failure_type<std::decay_t<EC>>{static_cast<EC &&>(v), spare_storage};
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class EC, class E> inline constexpr failure_type<std::decay_t<EC>, std::decay_t<E>> failure(EC &&v, E &&w, uint16_t spare_storage = 0)
{
  return failure_type<std::decay_t<EC>, std::decay_t<E>>{static_cast<EC &&>(v), static_cast<E &&>(w), spare_storage};
}
namespace detail
{
  template <class T> struct is_success_type
  {
    static constexpr bool value = false;
  };
  template <class T> struct is_success_type<success_type<T>>
  {
    static constexpr bool value = true;
  };
  template <class T> struct is_failure_type
  {
    static constexpr bool value = false;
  };
  template <class EC, class E> struct is_failure_type<failure_type<EC, E>>
  {
    static constexpr bool value = true;
  };
} // namespace detail
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> static constexpr bool is_success_type = detail::is_success_type<std::decay_t<T>>::value;
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> static constexpr bool is_failure_type = detail::is_failure_type<std::decay_t<T>>::value;
OUTCOME_V2_NAMESPACE_END
#endif
/* Traits for Outcome
(C) 2018-2019 Niall Douglas <http://www.nedproductions.biz/> (8 commits)
File Created: March 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_TRAIT_HPP
#define OUTCOME_TRAIT_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace trait
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R> //
  static constexpr bool type_can_be_used_in_basic_result = //
  (!std::is_reference<R>::value //
   && !OUTCOME_V2_NAMESPACE::detail::is_in_place_type_t<std::decay_t<R>>::value //
   && !is_success_type<R> //
   && !is_failure_type<R> //
   && !std::is_array<R>::value //
   && (std::is_void<R>::value || (std::is_object<R>::value //
                                  && std::is_destructible<R>::value)) //
  );
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_error_type. Potential doc page: NOT FOUND
*/
  template <class T> struct is_move_bitcopying
  {
    static constexpr bool value = false;
  };
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_error_type. Potential doc page: NOT FOUND
*/
  template <class E> struct is_error_type
  {
    static constexpr bool value = false;
  };
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_error_type_enum. Potential doc page: NOT FOUND
*/
  template <class E, class Enum> struct is_error_type_enum
  {
    static constexpr bool value = false;
  };
  namespace detail
  {
    template <class T> using devoid = OUTCOME_V2_NAMESPACE::detail::devoid<T>;
    template <class T> std::add_rvalue_reference_t<devoid<T>> declval() noexcept;
    // From http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4436.pdf
    namespace detector_impl
    {
      template <class...> using void_t = void;
      template <class Default, class, template <class...> class Op, class... Args> struct detector
      {
        static constexpr bool value = false;
        using type = Default;
      };
      template <class Default, template <class...> class Op, class... Args> struct detector<Default, void_t<Op<Args...>>, Op, Args...>
      {
        static constexpr bool value = true;
        using type = Op<Args...>;
      };
    } // namespace detector_impl
    template <template <class...> class Op, class... Args> using is_detected = detector_impl::detector<void, void, Op, Args...>;
    template <class Arg> using result_of_make_error_code = decltype(make_error_code(declval<Arg>()));
    template <class Arg> using introspect_make_error_code = is_detected<result_of_make_error_code, Arg>;
    template <class Arg> using result_of_make_exception_ptr = decltype(make_exception_ptr(declval<Arg>()));
    template <class Arg> using introspect_make_exception_ptr = is_detected<result_of_make_exception_ptr, Arg>;
    template <class T> struct _is_error_code_available
    {
      static constexpr bool value = detail::introspect_make_error_code<T>::value;
      using type = typename detail::introspect_make_error_code<T>::type;
    };
    template <class T> struct _is_exception_ptr_available
    {
      static constexpr bool value = detail::introspect_make_exception_ptr<T>::value;
      using type = typename detail::introspect_make_exception_ptr<T>::type;
    };
  } // namespace detail
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_error_code_available. Potential doc page: NOT FOUND
*/
  template <class T> struct is_error_code_available
  {
    static constexpr bool value = detail::_is_error_code_available<std::decay_t<T>>::value;
    using type = typename detail::_is_error_code_available<std::decay_t<T>>::type;
  };
  template <class T> constexpr bool is_error_code_available_v = detail::_is_error_code_available<std::decay_t<T>>::value;
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_exception_ptr_available. Potential doc page: NOT FOUND
*/
  template <class T> struct is_exception_ptr_available
  {
    static constexpr bool value = detail::_is_exception_ptr_available<std::decay_t<T>>::value;
    using type = typename detail::_is_exception_ptr_available<std::decay_t<T>>::type;
  };
  template <class T> constexpr bool is_exception_ptr_available_v = detail::_is_exception_ptr_available<std::decay_t<T>>::value;
} // namespace trait
OUTCOME_V2_NAMESPACE_END
#endif
/* Essentially an internal optional implementation :)
(C) 2017-2020 Niall Douglas <http://www.nedproductions.biz/> (24 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_VALUE_STORAGE_HPP
#define OUTCOME_VALUE_STORAGE_HPP
#include <cassert>
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class T, bool nothrow> struct strong_swap_impl
  {
    constexpr strong_swap_impl(bool &allgood, T &a, T &b)
    {
      allgood = true;
      using std::swap;
      swap(a, b);
    }
  };
  template <class T, bool nothrow> struct strong_placement_impl
  {
    template <class F> constexpr strong_placement_impl(bool &allgood, T *a, T *b, F &&f)
    {
      allgood = true;
      new(a) T(static_cast<T &&>(*b));
      b->~T();
      f();
    }
  };
#ifdef __cpp_exceptions
  template <class T> struct strong_swap_impl<T, false>
  {
    strong_swap_impl(bool &allgood, T &a, T &b)
    {
      allgood = true;
      T v(static_cast<T &&>(a));
      try
      {
        a = static_cast<T &&>(b);
      }
      catch(...)
      {
        // Try to put back a
        try
        {
          a = static_cast<T &&>(v);
          // fall through as all good
        }
        catch(...)
        {
          // failed to completely restore
          allgood = false;
          // throw away second exception
        }
        throw; // rethrow original exception
      }
      // b has been moved to a, try to move v to b
      try
      {
        b = static_cast<T &&>(v);
      }
      catch(...)
      {
        // Try to restore a to b, and v to a
        try
        {
          b = static_cast<T &&>(a);
          a = static_cast<T &&>(v);
          // fall through as all good
        }
        catch(...)
        {
          // failed to completely restore
          allgood = false;
          // throw away second exception
        }
        throw; // rethrow original exception
      }
    }
  };
  template <class T> struct strong_placement_impl<T, false>
  {
    template <class F> strong_placement_impl(bool &allgood, T *a, T *b, F &&f)
    {
      new(a) T(static_cast<T &&>(*b));
      try
      {
        b->~T();
        f();
      }
      catch(...)
      {
        // Try to put back a, but only if we are still good
        if(allgood)
        {
          try
          {
            new(b) T(static_cast<T &&>(*a));
            // fall through as all good
          }
          catch(...)
          {
            // failed to completely restore
            allgood = false;
            // throw away second exception
          }
          throw; // rethrow original exception
        }
      }
    }
  };
#endif
} // namespace detail
/*!
 */
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(std::is_move_constructible<T>::value &&std::is_move_assignable<T>::value))
constexpr inline void strong_swap(bool &allgood, T &a, T &b) noexcept(detail::is_nothrow_swappable<T>::value)
{
  detail::strong_swap_impl<T, detail::is_nothrow_swappable<T>::value>(allgood, a, b);
}
/*!
 */
OUTCOME_TEMPLATE(class T, class F)
OUTCOME_TREQUIRES(OUTCOME_TPRED(std::is_move_constructible<T>::value &&std::is_move_assignable<T>::value))
constexpr inline void strong_placement(bool &allgood, T *a, T *b, F &&f) noexcept(std::is_nothrow_move_constructible<T>::value)
{
  detail::strong_placement_impl<T, std::is_nothrow_move_constructible<T>::value>(allgood, a, b, static_cast<F &&>(f));
}
namespace detail
{
  template <class T>
  constexpr
#ifdef _MSC_VER
  __declspec(noreturn)
#elif defined(__GNUC__) || defined(__clang__)
        __attribute__((noreturn))
#endif
  void make_ub(T && /*unused*/)
  {
    assert(false); // NOLINT
#if defined(__GNUC__) || defined(__clang__)
    __builtin_unreachable();
#elif defined(_MSC_VER)
    __assume(0);
#endif
  }
  /* Outcome v1 used a C bitfield whose values were tracked by compiler optimisers nicely,
  but that produces ICEs when used in constexpr.

  Outcome v2.0-v2.1 used a 32 bit integer and manually set and cleared bits. Unfortunately
  only GCC's optimiser tracks bit values during constant folding, and only per byte, and
  even then unreliably. https://wg21.link/P1886 "Error speed benchmarking" showed just how
  poorly clang and MSVC fails to optimise outcome-using code, if you manually set bits.

  Outcome v2.2 therefore uses an enum with fixed values, and constexpr manipulation functions
  to change the value to one of the enum's values. This is stupid to look at in source code,
  but it make clang's optimiser do the right thing, so it's worth it.
  */
#define OUTCOME_USE_CONSTEXPR_ENUM_STATUS 0
  enum class status : uint16_t
  {
    // WARNING: These bits are not tracked by abi-dumper, but changing them will break ABI!
    none = 0,
    have_value = (1U << 0U),
    have_error = (1U << 1U),
    have_exception = (2U << 1U),
    have_error_exception = (3U << 1U),
    // failed to complete a strong swap
    have_lost_consistency = (1U << 3U),
    have_value_lost_consistency = (1U << 0U) | (1U << 3U),
    have_error_lost_consistency = (1U << 1U) | (1U << 3U),
    have_exception_lost_consistency = (2U << 1U) | (1U << 3U),
    have_error_exception_lost_consistency = (3U << 1U) | (1U << 3U),
    // can errno be set from this error?
    have_error_is_errno = (1U << 4U),
    have_error_error_is_errno = (1U << 1U) | (1U << 4U),
    have_error_exception_error_is_errno = (3U << 1U) | (1U << 4U),
    have_error_lost_consistency_error_is_errno = (1U << 1U) | (1U << 3U) | (1U << 4U),
    have_error_exception_lost_consistency_error_is_errno = (3U << 1U) | (1U << 3U) | (1U << 4U),
    // value has been moved from
    have_moved_from = (1U << 5U)
  };
  struct status_bitfield_type
  {
    status status_value{status::none};
    uint16_t spare_storage_value{0}; // hooks::spare_storage()
    constexpr status_bitfield_type() = default;
    constexpr status_bitfield_type(status v) noexcept
        : status_value(v)
    {
    } // NOLINT
    constexpr status_bitfield_type(status v, uint16_t s) noexcept
        : status_value(v)
        , spare_storage_value(s)
    {
    }
    constexpr status_bitfield_type(const status_bitfield_type &) = default;
    constexpr status_bitfield_type(status_bitfield_type &&) = default;
    constexpr status_bitfield_type &operator=(const status_bitfield_type &) = default;
    constexpr status_bitfield_type &operator=(status_bitfield_type &&) = default;
    //~status_bitfield_type() = default;  // Do NOT uncomment this, it breaks older clangs!
    constexpr bool have_value() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_value)) != 0;
    }
    constexpr bool have_error() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_error)) != 0;
    }
    constexpr bool have_exception() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_exception)) != 0;
    }
    constexpr bool have_lost_consistency() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_lost_consistency)) != 0;
    }
    constexpr bool have_error_is_errno() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_error_is_errno)) != 0;
    }
    constexpr bool have_moved_from() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_moved_from)) != 0;
    }
    constexpr status_bitfield_type &set_have_value(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_value)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_value)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_error(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_error)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_error)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_exception(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_exception)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_exception)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_error_is_errno(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_error_is_errno)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_error_is_errno)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_lost_consistency(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_lost_consistency)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_lost_consistency)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_moved_from(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_moved_from)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_moved_from)));
      return *this;
    }
  };
#if !defined(NDEBUG)
  // Check is trivial in all ways except default constructibility
  static_assert(sizeof(status_bitfield_type) == 4, "status_bitfield_type is not sized 4 bytes!");
  static_assert(std::is_trivially_copyable<status_bitfield_type>::value, "status_bitfield_type is not trivially copyable!");
  static_assert(std::is_trivially_assignable<status_bitfield_type, status_bitfield_type>::value, "status_bitfield_type is not trivially assignable!");
  static_assert(std::is_trivially_destructible<status_bitfield_type>::value, "status_bitfield_type is not trivially destructible!");
  static_assert(std::is_trivially_copy_constructible<status_bitfield_type>::value, "status_bitfield_type is not trivially copy constructible!");
  static_assert(std::is_trivially_move_constructible<status_bitfield_type>::value, "status_bitfield_type is not trivially move constructible!");
  static_assert(std::is_trivially_copy_assignable<status_bitfield_type>::value, "status_bitfield_type is not trivially copy assignable!");
  static_assert(std::is_trivially_move_assignable<status_bitfield_type>::value, "status_bitfield_type is not trivially move assignable!");
  // Also check is standard layout
  static_assert(std::is_standard_layout<status_bitfield_type>::value, "status_bitfield_type is not a standard layout type!");
#endif
  template <class State> constexpr inline void _set_error_is_errno(State & /*unused*/) {}
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4624) // destructor was implicitly defined as deleted
#endif
  // Used if both T and E are trivial
  template <class T, class E> struct value_storage_trivial
  {
    using value_type = T;
    using error_type = E;
    // Disable in place construction if they are the same type
    struct disable_in_place_value_type
    {
    };
    struct disable_in_place_error_type
    {
    };
    using _value_type = std::conditional_t<std::is_same<value_type, error_type>::value, disable_in_place_value_type, value_type>;
    using _error_type = std::conditional_t<std::is_same<value_type, error_type>::value, disable_in_place_error_type, error_type>;
    using _value_type_ = devoid<value_type>;
    using _error_type_ = devoid<error_type>;
    union {
      empty_type _empty;
      _value_type_ _value;
      _error_type_ _error;
    };
    status_bitfield_type _status;
    constexpr value_storage_trivial() noexcept
        : _empty{}
    {
    }
    value_storage_trivial(const value_storage_trivial &) = default; // NOLINT
    value_storage_trivial(value_storage_trivial &&) = default; // NOLINT
    value_storage_trivial &operator=(const value_storage_trivial &) = default; // NOLINT
    value_storage_trivial &operator=(value_storage_trivial &&) = default; // NOLINT
    ~value_storage_trivial() = default;
    constexpr explicit value_storage_trivial(status_bitfield_type status)
        : _empty()
        , _status(status)
    {
    }
    template <class... Args>
    constexpr explicit value_storage_trivial(in_place_type_t<_value_type> /*unused*/,
                                             Args &&... args) noexcept(detail::is_nothrow_constructible<_value_type_, Args...>)
        : _value(static_cast<Args &&>(args)...)
        , _status(status::have_value)
    {
    }
    template <class U, class... Args>
    constexpr value_storage_trivial(in_place_type_t<_value_type> /*unused*/, std::initializer_list<U> il,
                                    Args &&... args) noexcept(detail::is_nothrow_constructible<_value_type_, std::initializer_list<U>, Args...>)
        : _value(il, static_cast<Args &&>(args)...)
        , _status(status::have_value)
    {
    }
    template <class... Args>
    constexpr explicit value_storage_trivial(in_place_type_t<_error_type> /*unused*/,
                                             Args &&... args) noexcept(detail::is_nothrow_constructible<_error_type_, Args...>)
        : _error(static_cast<Args &&>(args)...)
        , _status(status::have_error)
    {
      _set_error_is_errno(*this);
    }
    template <class U, class... Args>
    constexpr value_storage_trivial(in_place_type_t<_error_type> /*unused*/, std::initializer_list<U> il,
                                    Args &&... args) noexcept(detail::is_nothrow_constructible<_error_type_, std::initializer_list<U>, Args...>)
        : _error(il, static_cast<Args &&>(args)...)
        , _status(status::have_error)
    {
      _set_error_is_errno(*this);
    }
    struct nonvoid_converting_constructor_tag
    {
    };
    template <class U, class V>
    static constexpr bool enable_nonvoid_converting_constructor =
    !(std::is_same<std::decay_t<U>, value_type>::value && std::is_same<std::decay_t<V>, error_type>::value) //
    && detail::is_constructible<value_type, U> && detail::is_constructible<error_type, V>;
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_trivial(const value_storage_trivial<U, V> &o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_trivial(o._status.have_value() ?
                                value_storage_trivial(in_place_type<value_type>, o._value) :
                                (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>, o._error) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_trivial(value_storage_trivial<U, V> &&o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_trivial(
          o._status.have_value() ?
          value_storage_trivial(in_place_type<value_type>, static_cast<U &&>(o._value)) :
          (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>, static_cast<V &&>(o._error)) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    struct void_value_converting_constructor_tag
    {
    };
    template <class V>
    static constexpr bool enable_void_value_converting_constructor =
    std::is_default_constructible<value_type>::value &&detail::is_constructible<error_type, V>;
    OUTCOME_TEMPLATE(class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_value_converting_constructor<V>))
    constexpr explicit value_storage_trivial(const value_storage_trivial<void, V> &o, void_value_converting_constructor_tag /*unused*/ = {}) noexcept(
    std::is_nothrow_default_constructible<_value_type_>::value &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_trivial(o._status.have_value() ?
                                value_storage_trivial(in_place_type<value_type>) :
                                (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>, o._error) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_value_converting_constructor<V>))
    constexpr explicit value_storage_trivial(value_storage_trivial<void, V> &&o, void_value_converting_constructor_tag /*unused*/ = {}) noexcept(
    std::is_nothrow_default_constructible<_value_type_>::value &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_trivial(
          o._status.have_value() ?
          value_storage_trivial(in_place_type<value_type>) :
          (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>, static_cast<V &&>(o._error)) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    struct void_error_converting_constructor_tag
    {
    };
    template <class U>
    static constexpr bool enable_void_error_converting_constructor =
    std::is_default_constructible<error_type>::value &&detail::is_constructible<value_type, U>;
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_error_converting_constructor<U>))
    constexpr explicit value_storage_trivial(const value_storage_trivial<U, void> &o, void_error_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&std::is_nothrow_default_constructible<_error_type_>::value)
        : value_storage_trivial(o._status.have_value() ?
                                value_storage_trivial(in_place_type<value_type>, o._value) :
                                (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_error_converting_constructor<U>))
    constexpr explicit value_storage_trivial(value_storage_trivial<U, void> &&o, void_error_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&std::is_nothrow_default_constructible<_error_type_>::value)
        : value_storage_trivial(o._status.have_value() ?
                                value_storage_trivial(in_place_type<value_type>, static_cast<U &&>(o._value)) :
                                (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    constexpr void swap(value_storage_trivial &o) noexcept
    {
      // storage is trivial, so just use assignment
      auto temp = static_cast<value_storage_trivial &&>(*this);
      *this = static_cast<value_storage_trivial &&>(o);
      o = static_cast<value_storage_trivial &&>(temp);
    }
  };
  // Used if T is non-trivial
  template <class T, class E> struct value_storage_nontrivial
  {
    using value_type = T;
    using error_type = E;
    struct disable_in_place_value_type
    {
    };
    struct disable_in_place_error_type
    {
    };
    using _value_type = std::conditional_t<std::is_same<value_type, error_type>::value, disable_in_place_value_type, value_type>;
    using _error_type = std::conditional_t<std::is_same<value_type, error_type>::value, disable_in_place_error_type, error_type>;
    using _value_type_ = devoid<value_type>;
    using _error_type_ = devoid<error_type>;
    union {
      empty_type _empty1;
      _value_type_ _value;
    };
    status_bitfield_type _status;
    union {
      empty_type _empty2;
      _error_type_ _error;
    };
    value_storage_nontrivial() noexcept
        : _empty1{}
        , _empty2{}
    {
    }
    value_storage_nontrivial &operator=(const value_storage_nontrivial &) = default; // if reaches here, copy assignment is trivial
    value_storage_nontrivial &operator=(value_storage_nontrivial &&) = default; // NOLINT if reaches here, move assignment is trivial
    value_storage_nontrivial(value_storage_nontrivial &&o) noexcept(
    std::is_nothrow_move_constructible<_value_type_>::value &&std::is_nothrow_move_constructible<_error_type_>::value) // NOLINT
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(static_cast<_value_type_ &&>(o._value)); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(static_cast<_error_type_ &&>(o._error)); // NOLINT
      }
      _status = o._status;
      o._status.set_have_moved_from(true);
    }
    value_storage_nontrivial(const value_storage_nontrivial &o) noexcept(
    std::is_nothrow_copy_constructible<_value_type_>::value &&std::is_nothrow_copy_constructible<_error_type_>::value)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(o._value); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(o._error); // NOLINT
      }
      _status = o._status;
    }
    explicit value_storage_nontrivial(status_bitfield_type status)
        : _empty1()
        , _status(status)
        , _empty2()
    {
    }
    template <class... Args>
    explicit value_storage_nontrivial(in_place_type_t<_value_type> /*unused*/,
                                      Args &&... args) noexcept(detail::is_nothrow_constructible<_value_type_, Args...>)
        : _value(static_cast<Args &&>(args)...) // NOLINT
        , _status(status::have_value)
    {
    }
    template <class U, class... Args>
    value_storage_nontrivial(in_place_type_t<_value_type> /*unused*/, std::initializer_list<U> il,
                             Args &&... args) noexcept(detail::is_nothrow_constructible<_value_type_, std::initializer_list<U>, Args...>)
        : _value(il, static_cast<Args &&>(args)...)
        , _status(status::have_value)
    {
    }
    template <class... Args>
    explicit value_storage_nontrivial(in_place_type_t<_error_type> /*unused*/,
                                      Args &&... args) noexcept(detail::is_nothrow_constructible<_error_type_, Args...>)
        : _status(status::have_error)
        , _error(static_cast<Args &&>(args)...) // NOLINT
    {
      _set_error_is_errno(*this);
    }
    template <class U, class... Args>
    value_storage_nontrivial(in_place_type_t<_error_type> /*unused*/, std::initializer_list<U> il,
                             Args &&... args) noexcept(detail::is_nothrow_constructible<_error_type_, std::initializer_list<U>, Args...>)
        : _status(status::have_error)
        , _error(il, static_cast<Args &&>(args)...)
    {
      _set_error_is_errno(*this);
    }
    struct nonvoid_converting_constructor_tag
    {
    };
    template <class U, class V>
    static constexpr bool enable_nonvoid_converting_constructor =
    !(std::is_same<std::decay_t<U>, value_type>::value && std::is_same<std::decay_t<V>, error_type>::value) //
    && detail::is_constructible<value_type, U> && detail::is_constructible<error_type, V>;
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_nontrivial(const value_storage_trivial<U, V> &o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_nontrivial(o._status.have_value() ?
                                   value_storage_nontrivial(in_place_type<value_type>, o._value) :
                                   (o._status.have_error() ? value_storage_nontrivial(in_place_type<error_type>, o._error) : value_storage_nontrivial()))
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_nontrivial(value_storage_trivial<U, V> &&o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_nontrivial(
          o._status.have_value() ?
          value_storage_nontrivial(in_place_type<value_type>, static_cast<U &&>(o._value)) :
          (o._status.have_error() ? value_storage_nontrivial(in_place_type<error_type>, static_cast<V &&>(o._error)) : value_storage_nontrivial()))
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_nontrivial(const value_storage_nontrivial<U, V> &o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_nontrivial(o._status.have_value() ?
                                   value_storage_nontrivial(in_place_type<value_type>, o._value) :
                                   (o._status.have_error() ? value_storage_nontrivial(in_place_type<error_type>, o._error) : value_storage_nontrivial()))
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_nontrivial(value_storage_nontrivial<U, V> &&o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_nontrivial(
          o._status.have_value() ?
          value_storage_nontrivial(in_place_type<value_type>, static_cast<U &&>(o._value)) :
          (o._status.have_error() ? value_storage_nontrivial(in_place_type<error_type>, static_cast<V &&>(o._error)) : value_storage_nontrivial()))
    {
      _status = o._status;
    }
    struct void_value_converting_constructor_tag
    {
    };
    template <class V>
    static constexpr bool enable_void_value_converting_constructor =
    std::is_default_constructible<value_type>::value &&detail::is_constructible<error_type, V>;
    OUTCOME_TEMPLATE(class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_value_converting_constructor<V>))
    explicit value_storage_nontrivial(const value_storage_trivial<void, V> &o, void_value_converting_constructor_tag /*unused*/ = {}) noexcept(
    std::is_nothrow_default_constructible<_value_type_>::value &&detail::is_nothrow_constructible<_error_type_, V>)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(o._error); // NOLINT
      }
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_value_converting_constructor<V>))
    explicit value_storage_nontrivial(value_storage_trivial<void, V> &&o, void_value_converting_constructor_tag /*unused*/ = {}) noexcept(
    std::is_nothrow_default_constructible<_value_type_>::value &&detail::is_nothrow_constructible<_error_type_, V>)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(static_cast<_error_type_ &&>(o._error)); // NOLINT
      }
      _status = o._status;
      o._status.set_have_moved_from(true);
    }
    struct void_error_converting_constructor_tag
    {
    };
    template <class U>
    static constexpr bool enable_void_error_converting_constructor =
    std::is_default_constructible<error_type>::value &&detail::is_constructible<value_type, U>;
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_error_converting_constructor<U>))
    explicit value_storage_nontrivial(const value_storage_trivial<U, void> &o, void_error_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&std::is_nothrow_default_constructible<_error_type_>::value)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(o._value); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(); // NOLINT
      }
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_error_converting_constructor<U>))
    explicit value_storage_nontrivial(value_storage_trivial<U, void> &&o, void_error_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&std::is_nothrow_default_constructible<_error_type_>::value)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(static_cast<_value_type_ &&>(o._value)); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(); // NOLINT
      }
      _status = o._status;
      o._status.set_have_moved_from(true);
    }
    ~value_storage_nontrivial() noexcept(std::is_nothrow_destructible<_value_type_>::value &&std::is_nothrow_destructible<_error_type_>::value)
    {
      if(this->_status.have_value())
      {
        if(!trait::is_move_bitcopying<value_type>::value || !this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        this->_status.set_have_value(false);
      }
      else if(this->_status.have_error())
      {
        if(!trait::is_move_bitcopying<error_type>::value || !this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        this->_status.set_have_error(false);
      }
    }
    constexpr void
    swap(value_storage_nontrivial &o) noexcept(detail::is_nothrow_swappable<_value_type_>::value &&detail::is_nothrow_swappable<_error_type_>::value)
    {
      using std::swap;
      // empty/empty
      if(!_status.have_value() && !o._status.have_value() && !_status.have_error() && !o._status.have_error())
      {
        swap(_status, o._status);
        return;
      }
      // value/value
      if(_status.have_value() && o._status.have_value())
      {
        struct _
        {
          status_bitfield_type &a, &b;
          bool all_good{false};
          ~_()
          {
            if(!all_good)
            {
              // We lost one of the values
              a.set_have_lost_consistency(true);
              b.set_have_lost_consistency(true);
            }
          }
        } _{_status, o._status};
        strong_swap(_.all_good, _value, o._value);
        swap(_status, o._status);
        return;
      }
      // error/error
      if(_status.have_error() && o._status.have_error())
      {
        struct _
        {
          status_bitfield_type &a, &b;
          bool all_good{false};
          ~_()
          {
            if(!all_good)
            {
              // We lost one of the values
              a.set_have_lost_consistency(true);
              b.set_have_lost_consistency(true);
            }
          }
        } _{_status, o._status};
        strong_swap(_.all_good, _error, o._error);
        swap(_status, o._status);
        return;
      }
      // Could be value/empty, error/empty, etc
      if(_status.have_value() && !o._status.have_error())
      {
        // Move construct me into other
        new(&o._value) _value_type_(static_cast<_value_type_ &&>(_value)); // NOLINT
        if(!trait::is_move_bitcopying<value_type>::value)
        {
          this->_value.~value_type(); // NOLINT
        }
        swap(_status, o._status);
        return;
      }
      if(o._status.have_value() && !_status.have_error())
      {
        // Move construct other into me
        new(&_value) _value_type_(static_cast<_value_type_ &&>(o._value)); // NOLINT
        if(!trait::is_move_bitcopying<value_type>::value)
        {
          o._value.~value_type(); // NOLINT
        }
        swap(_status, o._status);
        return;
      }
      if(_status.have_error() && !o._status.have_value())
      {
        // Move construct me into other
        new(&o._error) _error_type_(static_cast<_error_type_ &&>(_error)); // NOLINT
        if(!trait::is_move_bitcopying<error_type>::value)
        {
          this->_error.~error_type(); // NOLINT
        }
        swap(_status, o._status);
        return;
      }
      if(o._status.have_error() && !_status.have_value())
      {
        // Move construct other into me
        new(&_error) _error_type_(static_cast<_error_type_ &&>(o._error)); // NOLINT
        if(!trait::is_move_bitcopying<error_type>::value)
        {
          o._error.~error_type(); // NOLINT
        }
        swap(_status, o._status);
        return;
      }
      // It can now only be value/error, or error/value
      struct _
      {
        status_bitfield_type &a, &b;
        _value_type_ *value, *o_value;
        _error_type_ *error, *o_error;
        bool all_good{true};
        ~_()
        {
          if(!all_good)
          {
            // We lost one of the values
            a.set_have_lost_consistency(true);
            b.set_have_lost_consistency(true);
          }
        }
      } _{_status, o._status, &_value, &o._value, &_error, &o._error};
      if(_status.have_value() && o._status.have_error())
      {
        strong_placement(_.all_good, _.o_value, _.value, [&_] { //
          strong_placement(_.all_good, _.error, _.o_error, [&_] { //
            swap(_.a, _.b); //
          });
        });
        return;
      }
      if(_status.have_error() && o._status.have_value())
      {
        strong_placement(_.all_good, _.o_error, _.error, [&_] { //
          strong_placement(_.all_good, _.value, _.o_value, [&_] { //
            swap(_.a, _.b); //
          });
        });
        return;
      }
      // Should never reach here
      make_ub(_value);
    }
  };
  template <class Base> struct value_storage_delete_copy_constructor : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_delete_copy_constructor() = default;
    value_storage_delete_copy_constructor(const value_storage_delete_copy_constructor &) = delete;
    value_storage_delete_copy_constructor(value_storage_delete_copy_constructor &&) = default; // NOLINT
  };
  template <class Base> struct value_storage_delete_copy_assignment : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_delete_copy_assignment() = default;
    value_storage_delete_copy_assignment(const value_storage_delete_copy_assignment &) = default;
    value_storage_delete_copy_assignment(value_storage_delete_copy_assignment &&) = default; // NOLINT
    value_storage_delete_copy_assignment &operator=(const value_storage_delete_copy_assignment &o) = delete;
    value_storage_delete_copy_assignment &operator=(value_storage_delete_copy_assignment &&o) = default; // NOLINT
  };
  template <class Base> struct value_storage_delete_move_assignment : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_delete_move_assignment() = default;
    value_storage_delete_move_assignment(const value_storage_delete_move_assignment &) = default;
    value_storage_delete_move_assignment(value_storage_delete_move_assignment &&) = default; // NOLINT
    value_storage_delete_move_assignment &operator=(const value_storage_delete_move_assignment &o) = default;
    value_storage_delete_move_assignment &operator=(value_storage_delete_move_assignment &&o) = delete;
  };
  template <class Base> struct value_storage_delete_move_constructor : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_delete_move_constructor() = default;
    value_storage_delete_move_constructor(const value_storage_delete_move_constructor &) = default;
    value_storage_delete_move_constructor(value_storage_delete_move_constructor &&) = delete;
  };
  template <class Base> struct value_storage_nontrivial_move_assignment : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_nontrivial_move_assignment() = default;
    value_storage_nontrivial_move_assignment(const value_storage_nontrivial_move_assignment &) = default;
    value_storage_nontrivial_move_assignment(value_storage_nontrivial_move_assignment &&) = default; // NOLINT
    value_storage_nontrivial_move_assignment &operator=(const value_storage_nontrivial_move_assignment &o) = default;
    value_storage_nontrivial_move_assignment &operator=(value_storage_nontrivial_move_assignment &&o) noexcept(
    std::is_nothrow_move_assignable<value_type>::value &&std::is_nothrow_move_assignable<error_type>::value) // NOLINT
    {
      using _value_type_ = typename Base::_value_type_;
      using _error_type_ = typename Base::_error_type_;
      if(!this->_status.have_value() && !this->_status.have_error() && !o._status.have_value() && !o._status.have_error())
      {
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_value() && o._status.have_value())
      {
        this->_value = static_cast<_value_type_&&>(o._value); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_error() && o._status.have_error())
      {
        this->_error = static_cast<_error_type_&&>(o._error); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_value() && !o._status.have_value() && !o._status.have_error())
      {
        if(!trait::is_move_bitcopying<value_type>::value || this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(!this->_status.have_value() && !this->_status.have_error() && o._status.have_value())
      {
        new(&this->_value) _value_type_(static_cast<_value_type_&&>(o._value)); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_error() && !o._status.have_value() && !o._status.have_error())
      {
        if(!trait::is_move_bitcopying<error_type>::value || this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(!this->_status.have_value() && !this->_status.have_error() && o._status.have_error())
      {
        new(&this->_error) _error_type_(static_cast<_error_type_&&>(o._error)); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_value() && o._status.have_error())
      {
        if(!trait::is_move_bitcopying<value_type>::value || this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        new(&this->_error) _error_type_(static_cast<_error_type_&&>(o._error)); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_error() && o._status.have_value())
      {
        if(!trait::is_move_bitcopying<error_type>::value || this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        new(&this->_value) _value_type_(static_cast<_value_type_&&>(o._value)); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      // Should never reach here
      make_ub(this->_value);
    }
  };
  template <class Base> struct value_storage_nontrivial_copy_assignment : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_nontrivial_copy_assignment() = default;
    value_storage_nontrivial_copy_assignment(const value_storage_nontrivial_copy_assignment &) = default;
    value_storage_nontrivial_copy_assignment(value_storage_nontrivial_copy_assignment &&) = default; // NOLINT
    value_storage_nontrivial_copy_assignment &operator=(value_storage_nontrivial_copy_assignment &&o) = default; // NOLINT
    value_storage_nontrivial_copy_assignment &operator=(const value_storage_nontrivial_copy_assignment &o) noexcept(
    std::is_nothrow_copy_assignable<value_type>::value &&std::is_nothrow_copy_assignable<error_type>::value)
    {
      using _value_type_ = typename Base::_value_type_;
      using _error_type_ = typename Base::_error_type_;
      if(!this->_status.have_value() && !this->_status.have_error() && !o._status.have_value()
         && !o._status.have_error())
      {
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_value() && o._status.have_value())
      {
        this->_value = o._value; // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_error() && o._status.have_error())
      {
        this->_error = o._error; // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_value() && !o._status.have_value() && !o._status.have_error())
      {
        if(!trait::is_move_bitcopying<value_type>::value || this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        this->_status = o._status;
        return *this;
      }
      if(!this->_status.have_value() && !this->_status.have_error() && o._status.have_value())
      {
        new(&this->_value) _value_type_(o._value); // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_error() && !o._status.have_value() && !o._status.have_error())
      {
        if(!trait::is_move_bitcopying<error_type>::value || this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        this->_status = o._status;
        return *this;
      }
      if(!this->_status.have_value() && !this->_status.have_error() && o._status.have_error())
      {
        new(&this->_error) _error_type_(o._error); // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_value() && o._status.have_error())
      {
        if(!trait::is_move_bitcopying<value_type>::value || this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        new(&this->_error) _error_type_(o._error); // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_error() && o._status.have_value())
      {
        if(!trait::is_move_bitcopying<error_type>::value || this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        new(&this->_value) _value_type_(o._value); // NOLINT
        this->_status = o._status;
        return *this;
      }
      // Should never reach here
      make_ub(this->_value);
    }
  };
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  // is_trivially_copyable is true even if type is not copyable, so handle that here
  template <class T> struct is_storage_trivial
  {
    static constexpr bool value = std::is_void<T>::value || (std::is_trivially_copy_constructible<T>::value && std::is_trivially_copyable<T>::value);
  };
  // work around libstdc++ 7 bug
  template <> struct is_storage_trivial<void>
  {
    static constexpr bool value = true;
  };
  template <> struct is_storage_trivial<const void>
  {
    static constexpr bool value = true;
  };
  template <class T, class E>
  using value_storage_select_trivality =
  std::conditional_t<is_storage_trivial<T>::value && is_storage_trivial<E>::value, value_storage_trivial<T, E>, value_storage_nontrivial<T, E>>;
  template <class T, class E>
  using value_storage_select_move_constructor =
  std::conditional_t<std::is_move_constructible<devoid<T>>::value && std::is_move_constructible<devoid<E>>::value, value_storage_select_trivality<T, E>,
                     value_storage_delete_move_constructor<value_storage_select_trivality<T, E>>>;
  template <class T, class E>
  using value_storage_select_copy_constructor =
  std::conditional_t<std::is_copy_constructible<devoid<T>>::value && std::is_copy_constructible<devoid<E>>::value, value_storage_select_move_constructor<T, E>,
                     value_storage_delete_copy_constructor<value_storage_select_move_constructor<T, E>>>;
  template <class T, class E>
  using value_storage_select_move_assignment =
  std::conditional_t<std::is_trivially_move_assignable<devoid<T>>::value && std::is_trivially_move_assignable<devoid<E>>::value,
                     value_storage_select_copy_constructor<T, E>,
                     std::conditional_t<std::is_move_assignable<devoid<T>>::value && std::is_move_assignable<devoid<E>>::value,
                                        value_storage_nontrivial_move_assignment<value_storage_select_copy_constructor<T, E>>,
                                        value_storage_delete_copy_assignment<value_storage_select_copy_constructor<T, E>>>>;
  template <class T, class E>
  using value_storage_select_copy_assignment =
  std::conditional_t<std::is_trivially_copy_assignable<devoid<T>>::value && std::is_trivially_copy_assignable<devoid<E>>::value,
                     value_storage_select_move_assignment<T, E>,
                     std::conditional_t<std::is_copy_assignable<devoid<T>>::value && std::is_copy_assignable<devoid<E>>::value,
                                        value_storage_nontrivial_copy_assignment<value_storage_select_move_assignment<T, E>>,
                                        value_storage_delete_copy_assignment<value_storage_select_move_assignment<T, E>>>>;
  template <class T, class E> using value_storage_select_impl = value_storage_select_copy_assignment<T, E>;
#ifndef NDEBUG
  // Check is trivial in all ways except default constructibility
  // static_assert(std::is_trivial<value_storage_select_impl<int, long>>::value, "value_storage_select_impl<int, long> is not trivial!");
  // static_assert(std::is_trivially_default_constructible<value_storage_select_impl<int, long>>::value, "value_storage_select_impl<int, long> is not trivially
  // default constructible!");
  static_assert(std::is_trivially_copyable<value_storage_select_impl<int, long>>::value, "value_storage_select_impl<int, long> is not trivially copyable!");
  static_assert(std::is_trivially_assignable<value_storage_select_impl<int, long>, value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially assignable!");
  static_assert(std::is_trivially_destructible<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially destructible!");
  static_assert(std::is_trivially_copy_constructible<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially copy constructible!");
  static_assert(std::is_trivially_move_constructible<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially move constructible!");
  static_assert(std::is_trivially_copy_assignable<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially copy assignable!");
  static_assert(std::is_trivially_move_assignable<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially move assignable!");
  // Also check is standard layout
  static_assert(std::is_standard_layout<value_storage_select_impl<int, long>>::value, "value_storage_select_impl<int, long> is not a standard layout type!");
#endif
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class R, class EC, class NoValuePolicy> class basic_result_storage;
} // namespace detail
namespace hooks
{
  template <class R, class S, class NoValuePolicy> constexpr inline uint16_t spare_storage(const detail::basic_result_storage<R, S, NoValuePolicy> *r) noexcept;
  template <class R, class S, class NoValuePolicy>
  constexpr inline void set_spare_storage(detail::basic_result_storage<R, S, NoValuePolicy> *r, uint16_t v) noexcept;
} // namespace hooks
namespace policy
{
  struct base;
} // namespace policy
namespace detail
{
  template <class R, class EC, class NoValuePolicy> //
  class basic_result_storage
  {
    static_assert(trait::type_can_be_used_in_basic_result<R>, "The type R cannot be used in a basic_result");
    static_assert(trait::type_can_be_used_in_basic_result<EC>, "The type S cannot be used in a basic_result");
    friend struct policy::base;
    template <class T, class U, class V> //
    friend class basic_result_storage;
    template <class T, class U, class V> friend class basic_result_final;
    template <class T, class U, class V>
    friend constexpr inline uint16_t hooks::spare_storage(const detail::basic_result_storage<T, U, V> *r) noexcept; // NOLINT
    template <class T, class U, class V>
    friend constexpr inline void hooks::set_spare_storage(detail::basic_result_storage<T, U, V> *r, uint16_t v) noexcept; // NOLINT
    struct disable_in_place_value_type
    {
    };
    struct disable_in_place_error_type
    {
    };
  protected:
    using _value_type = std::conditional_t<std::is_same<R, EC>::value, disable_in_place_value_type, R>;
    using _error_type = std::conditional_t<std::is_same<R, EC>::value, disable_in_place_error_type, EC>;
    using _state_type = value_storage_select_impl<_value_type, _error_type>;
    _state_type _state;
  public:
    // Used by iostream support to access state
    _state_type &_iostreams_state() { return _state; }
    const _state_type &_iostreams_state() const { return _state; }
  protected:
    basic_result_storage() = default;
    basic_result_storage(const basic_result_storage &) = default; // NOLINT
    basic_result_storage(basic_result_storage &&) = default; // NOLINT
    basic_result_storage &operator=(const basic_result_storage &) = default; // NOLINT
    basic_result_storage &operator=(basic_result_storage &&) = default; // NOLINT
    ~basic_result_storage() = default;
    template <class... Args>
    constexpr explicit basic_result_storage(in_place_type_t<_value_type> _,
                                            Args &&... args) noexcept(detail::is_nothrow_constructible<_value_type, Args...>)
        : _state{_, static_cast<Args &&>(args)...}
    {
    }
    template <class U, class... Args>
    constexpr basic_result_storage(in_place_type_t<_value_type> _, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<_value_type, std::initializer_list<U>, Args...>)
        : _state{_, il, static_cast<Args &&>(args)...}
    {
    }
    template <class... Args>
    constexpr explicit basic_result_storage(in_place_type_t<_error_type> _,
                                            Args &&... args) noexcept(detail::is_nothrow_constructible<_error_type, Args...>)
        : _state{_, static_cast<Args &&>(args)...}
    {
    }
    template <class U, class... Args>
    constexpr basic_result_storage(in_place_type_t<_error_type> _, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<_error_type, std::initializer_list<U>, Args...>)
        : _state{_, il, static_cast<Args &&>(args)...}
    {
    }
    struct compatible_conversion_tag
    {
    };
    template <class T, class U, class V>
    constexpr basic_result_storage(compatible_conversion_tag /*unused*/, const basic_result_storage<T, U, V> &o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&detail::is_nothrow_constructible<_error_type, U>)
        : _state(o._state)
    {
    }
    template <class T, class U, class V>
    constexpr basic_result_storage(compatible_conversion_tag /*unused*/, basic_result_storage<T, U, V> &&o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&detail::is_nothrow_constructible<_error_type, U>)
        : _state(static_cast<decltype(o._state) &&>(o._state))
    {
    }
    struct make_error_code_compatible_conversion_tag
    {
    };
    template <class T, class U, class V>
    constexpr basic_result_storage(make_error_code_compatible_conversion_tag /*unused*/, const basic_result_storage<T, U, V> &o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&noexcept(make_error_code(std::declval<U>())))
        : _state(o._state._status.have_value() ? _state_type(in_place_type<_value_type>, o._state._value) :
                                                 _state_type(in_place_type<_error_type>, make_error_code(o._state._error)))
    {
    }
    template <class T, class U, class V>
    constexpr basic_result_storage(make_error_code_compatible_conversion_tag /*unused*/, basic_result_storage<T, U, V> &&o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&noexcept(make_error_code(std::declval<U>())))
        : _state(o._state._status.have_value() ? _state_type(in_place_type<_value_type>, static_cast<T &&>(o._state._value)) :
                                                 _state_type(in_place_type<_error_type>, make_error_code(static_cast<U &&>(o._state._error))))
    {
    }
    struct make_exception_ptr_compatible_conversion_tag
    {
    };
    template <class T, class U, class V>
    constexpr basic_result_storage(make_exception_ptr_compatible_conversion_tag /*unused*/, const basic_result_storage<T, U, V> &o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&noexcept(make_exception_ptr(std::declval<U>())))
        : _state(o._state._status.have_value() ? _state_type(in_place_type<_value_type>, o._state._value) :
                                                 _state_type(in_place_type<_error_type>, make_exception_ptr(o._state._error)))
    {
    }
    template <class T, class U, class V>
    constexpr basic_result_storage(make_exception_ptr_compatible_conversion_tag /*unused*/, basic_result_storage<T, U, V> &&o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&noexcept(make_exception_ptr(std::declval<U>())))
        : _state(o._state._status.have_value() ? _state_type(in_place_type<_value_type>, static_cast<T &&>(o._state._value)) :
                                                 _state_type(in_place_type<_error_type>, make_exception_ptr(static_cast<U &&>(o._state._error))))
    {
    }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace concepts
{
#if defined(__cpp_concepts)
#if (defined(_MSC_VER) || defined(__clang__) || (defined(__GNUC__) && __cpp_concepts >= 201707) || OUTCOME_FORCE_STD_CXX_CONCEPTS) && !OUTCOME_FORCE_LEGACY_GCC_CXX_CONCEPTS
#define OUTCOME_GCC6_CONCEPT_BOOL
#else
#define OUTCOME_GCC6_CONCEPT_BOOL bool
#endif
  namespace detail
  {
    template <class T, class U> concept OUTCOME_GCC6_CONCEPT_BOOL SameHelper = std::is_same<T, U>::value;
    template <class T, class U> concept OUTCOME_GCC6_CONCEPT_BOOL same_as = detail::SameHelper<T, U> &&detail::SameHelper<U, T>;
    template <class T, class U> concept OUTCOME_GCC6_CONCEPT_BOOL convertible = std::is_convertible<T, U>::value;
    template <class T, class U> concept OUTCOME_GCC6_CONCEPT_BOOL base_of = std::is_base_of<T, U>::value;
  } // namespace detail
  /* The `value_or_none` concept.
  \requires That `U::value_type` exists and that `std::declval<U>().has_value()` returns a `bool` and `std::declval<U>().value()` exists.
  */
  template <class U> concept OUTCOME_GCC6_CONCEPT_BOOL value_or_none = requires(U a)
  {
    {
      a.has_value()
    }
    ->detail::same_as<bool>;
    {a.value()};
  };
  /* The `value_or_error` concept.
  \requires That `U::value_type` and `U::error_type` exist;
  that `std::declval<U>().has_value()` returns a `bool`, `std::declval<U>().value()` and  `std::declval<U>().error()` exists.
  */
  template <class U> concept OUTCOME_GCC6_CONCEPT_BOOL value_or_error = requires(U a)
  {
    {
      a.has_value()
    }
    ->detail::same_as<bool>;
    {a.value()};
    {a.error()};
  };
#else
  namespace detail
  {
    struct no_match
    {
    };
    inline no_match match_value_or_none(...);
    inline no_match match_value_or_error(...);
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<U>().has_value()), OUTCOME_TEXPR(std::declval<U>().value()))
    inline U match_value_or_none(U &&);
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<U>().has_value()), OUTCOME_TEXPR(std::declval<U>().value()), OUTCOME_TEXPR(std::declval<U>().error()))
    inline U match_value_or_error(U &&);
    template <class U>
    static constexpr bool value_or_none =
    !std::is_same<no_match, decltype(match_value_or_none(std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
    template <class U>
    static constexpr bool value_or_error =
    !std::is_same<no_match, decltype(match_value_or_error(std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
  } // namespace detail
  /* The `value_or_none` concept.
  \requires That `U::value_type` exists and that `std::declval<U>().has_value()` returns a `bool` and `std::declval<U>().value()` exists.
  */
  template <class U> static constexpr bool value_or_none = detail::value_or_none<U>;
  /* The `value_or_error` concept.
  \requires That `U::value_type` and `U::error_type` exist;
  that `std::declval<U>().has_value()` returns a `bool`, `std::declval<U>().value()` and  `std::declval<U>().error()` exists.
  */
  template <class U> static constexpr bool value_or_error = detail::value_or_error<U>;
#endif
} // namespace concepts
namespace convert
{
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
#if defined(__cpp_concepts)
  template <class U> concept OUTCOME_GCC6_CONCEPT_BOOL ValueOrNone = concepts::value_or_none<U>;
  template <class U> concept OUTCOME_GCC6_CONCEPT_BOOL ValueOrError = concepts::value_or_error<U>;
#else
  template <class U> static constexpr bool ValueOrNone = concepts::value_or_none<U>;
  template <class U> static constexpr bool ValueOrError = concepts::value_or_error<U>;
#endif
#endif
  namespace detail
  {
    template <class T, class X> struct make_type
    {
      template <class U> static constexpr T value(U &&v) { return T{in_place_type<typename T::value_type>, static_cast<U &&>(v).value()}; }
      template <class U> static constexpr T error(U &&v) { return T{in_place_type<typename T::error_type>, static_cast<U &&>(v).error()}; }
      static constexpr T error() { return T{in_place_type<typename T::error_type>}; }
    };
    template <class T> struct make_type<T, void>
    {
      template <class U> static constexpr T value(U && /*unused*/) { return T{in_place_type<typename T::value_type>}; }
      template <class U> static constexpr T error(U && /*unused*/) { return T{in_place_type<typename T::error_type>}; }
      static constexpr T error() { return T{in_place_type<typename T::error_type>}; }
    };
  } // namespace detail
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  value_or_error. Potential doc page: NOT FOUND
*/
  template <class T, class U> struct value_or_error
  {
    static constexpr bool enable_result_inputs = false;
    static constexpr bool enable_outcome_inputs = false;
    OUTCOME_TEMPLATE(class X)
    OUTCOME_TREQUIRES(
    OUTCOME_TPRED(std::is_same<U, std::decay_t<X>>::value //
                  &&concepts::value_or_error<U> //
                  && (std::is_void<typename std::decay_t<X>::value_type>::value ||
                      OUTCOME_V2_NAMESPACE::detail::is_explicitly_constructible<typename T::value_type, typename std::decay_t<X>::value_type>) //
                  &&(std::is_void<typename std::decay_t<X>::error_type>::value ||
                     OUTCOME_V2_NAMESPACE::detail::is_explicitly_constructible<typename T::error_type, typename std::decay_t<X>::error_type>) ))
    constexpr T operator()(X &&v)
    {
      return v.has_value() ? detail::make_type<T, typename T::value_type>::value(static_cast<X &&>(v)) :
                             detail::make_type<T, typename U::error_type>::error(static_cast<X &&>(v));
    }
  };
} // namespace convert
OUTCOME_V2_NAMESPACE_END
#endif
/* Finaliser for a very simple result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_RESULT_FINAL_HPP
#define OUTCOME_BASIC_RESULT_FINAL_HPP
/* Error observers for a very simple basic_result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (2 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_RESULT_ERROR_OBSERVERS_HPP
#define OUTCOME_BASIC_RESULT_ERROR_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class Base, class EC, class NoValuePolicy> class basic_result_error_observers : public Base
  {
  public:
    using error_type = EC;
    using Base::Base;
    constexpr error_type &assume_error() & noexcept
    {
      NoValuePolicy::narrow_error_check(static_cast<basic_result_error_observers &>(*this));
      return this->_state._error;
    }
    constexpr const error_type &assume_error() const &noexcept
    {
      NoValuePolicy::narrow_error_check(static_cast<const basic_result_error_observers &>(*this));
      return this->_state._error;
    }
    constexpr error_type &&assume_error() && noexcept
    {
      NoValuePolicy::narrow_error_check(static_cast<basic_result_error_observers &&>(*this));
      return static_cast<error_type &&>(this->_state._error);
    }
    constexpr const error_type &&assume_error() const &&noexcept
    {
      NoValuePolicy::narrow_error_check(static_cast<const basic_result_error_observers &&>(*this));
      return static_cast<const error_type &&>(this->_state._error);
    }
    constexpr error_type &error() &
    {
      NoValuePolicy::wide_error_check(static_cast<basic_result_error_observers &>(*this));
      return this->_state._error;
    }
    constexpr const error_type &error() const &
    {
      NoValuePolicy::wide_error_check(static_cast<const basic_result_error_observers &>(*this));
      return this->_state._error;
    }
    constexpr error_type &&error() &&
    {
      NoValuePolicy::wide_error_check(static_cast<basic_result_error_observers &&>(*this));
      return static_cast<error_type &&>(this->_state._error);
    }
    constexpr const error_type &&error() const &&
    {
      NoValuePolicy::wide_error_check(static_cast<const basic_result_error_observers &&>(*this));
      return static_cast<const error_type &&>(this->_state._error);
    }
  };
  template <class Base, class NoValuePolicy> class basic_result_error_observers<Base, void, NoValuePolicy> : public Base
  {
  public:
    using Base::Base;
    constexpr void assume_error() const noexcept { NoValuePolicy::narrow_error_check(*this); }
    constexpr void error() const { NoValuePolicy::wide_error_check(*this); }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
/* Value observers for a very simple basic_result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (2 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_RESULT_VALUE_OBSERVERS_HPP
#define OUTCOME_RESULT_VALUE_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class Base, class R, class NoValuePolicy> class basic_result_value_observers : public Base
  {
  public:
    using value_type = R;
    using Base::Base;
    constexpr value_type &assume_value() & noexcept
    {
      NoValuePolicy::narrow_value_check(static_cast<basic_result_value_observers &>(*this));
      return this->_state._value; // NOLINT
    }
    constexpr const value_type &assume_value() const &noexcept
    {
      NoValuePolicy::narrow_value_check(static_cast<const basic_result_value_observers &>(*this));
      return this->_state._value; // NOLINT
    }
    constexpr value_type &&assume_value() && noexcept
    {
      NoValuePolicy::narrow_value_check(static_cast<basic_result_value_observers &&>(*this));
      return static_cast<value_type &&>(this->_state._value); // NOLINT
    }
    constexpr const value_type &&assume_value() const &&noexcept
    {
      NoValuePolicy::narrow_value_check(static_cast<const basic_result_value_observers &&>(*this));
      return static_cast<const value_type &&>(this->_state._value); // NOLINT
    }
    constexpr value_type &value() &
    {
      NoValuePolicy::wide_value_check(static_cast<basic_result_value_observers &>(*this));
      return this->_state._value; // NOLINT
    }
    constexpr const value_type &value() const &
    {
      NoValuePolicy::wide_value_check(static_cast<const basic_result_value_observers &>(*this));
      return this->_state._value; // NOLINT
    }
    constexpr value_type &&value() &&
    {
      NoValuePolicy::wide_value_check(static_cast<basic_result_value_observers &&>(*this));
      return static_cast<value_type &&>(this->_state._value); // NOLINT
    }
    constexpr const value_type &&value() const &&
    {
      NoValuePolicy::wide_value_check(static_cast<const basic_result_value_observers &&>(*this));
      return static_cast<const value_type &&>(this->_state._value); // NOLINT
    }
  };
  template <class Base, class NoValuePolicy> class basic_result_value_observers<Base, void, NoValuePolicy> : public Base
  {
  public:
    using Base::Base;
    constexpr void assume_value() const noexcept { NoValuePolicy::narrow_value_check(*this); }
    constexpr void value() const { NoValuePolicy::wide_value_check(*this); }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class R, class EC, class NoValuePolicy> using select_basic_result_impl = basic_result_error_observers<basic_result_value_observers<basic_result_storage<R, EC, NoValuePolicy>, R, NoValuePolicy>, EC, NoValuePolicy>;
  template <class R, class S, class NoValuePolicy>
  class basic_result_final
  : public select_basic_result_impl<R, S, NoValuePolicy>
  {
    using base = select_basic_result_impl<R, S, NoValuePolicy>;
  public:
    using base::base;
    constexpr explicit operator bool() const noexcept { return this->_state._status.have_value(); }
    constexpr bool has_value() const noexcept { return this->_state._status.have_value(); }
    constexpr bool has_error() const noexcept { return this->_state._status.have_error(); }
    constexpr bool has_exception() const noexcept { return this->_state._status.have_exception(); }
    constexpr bool has_lost_consistency() const noexcept { return this->_state._status.have_lost_consistency(); }
    constexpr bool has_failure() const noexcept { return this->_state._status.have_error() || this->_state._status.have_exception(); }
    OUTCOME_TEMPLATE(class T, class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<detail::devoid<R>>() == std::declval<detail::devoid<T>>()), //
                      OUTCOME_TEXPR(std::declval<detail::devoid<S>>() == std::declval<detail::devoid<U>>()))
    constexpr bool operator==(const basic_result_final<T, U, V> &o) const noexcept( //
    noexcept(std::declval<detail::devoid<R>>() == std::declval<detail::devoid<T>>()) && noexcept(std::declval<detail::devoid<S>>() == std::declval<detail::devoid<U>>()))
    {
      if(this->_state._status.have_value() && o._state._status.have_value())
      {
        return this->_state._value == o._state._value; // NOLINT
      }
      if(this->_state._status.have_error() && o._state._status.have_error())
      {
        return this->_state._error == o._state._error;
      }
      return false;
    }
    OUTCOME_TEMPLATE(class T)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<R>() == std::declval<T>()))
    constexpr bool operator==(const success_type<T> &o) const noexcept( //
    noexcept(std::declval<R>() == std::declval<T>()))
    {
      if(this->_state._status.have_value())
      {
        return this->_state._value == o.value();
      }
      return false;
    }
    constexpr bool operator==(const success_type<void> &o) const noexcept
    {
      (void) o;
      return this->_state._status.have_value();
    }
    OUTCOME_TEMPLATE(class T)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<S>() == std::declval<T>()))
    constexpr bool operator==(const failure_type<T, void> &o) const noexcept( //
    noexcept(std::declval<S>() == std::declval<T>()))
    {
      if(this->_state._status.have_error())
      {
        return this->_state._error == o.error();
      }
      return false;
    }
    OUTCOME_TEMPLATE(class T, class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<detail::devoid<R>>() != std::declval<detail::devoid<T>>()), //
                      OUTCOME_TEXPR(std::declval<detail::devoid<S>>() != std::declval<detail::devoid<U>>()))
    constexpr bool operator!=(const basic_result_final<T, U, V> &o) const noexcept( //
    noexcept(std::declval<detail::devoid<R>>() != std::declval<detail::devoid<T>>()) && noexcept(std::declval<detail::devoid<S>>() != std::declval<detail::devoid<U>>()))
    {
      if(this->_state._status.have_value() && o._state._status.have_value())
      {
        return this->_state._value != o._state._value;
      }
      if(this->_state._status.have_error() && o._state._status.have_error())
      {
        return this->_state._error != o._state._error;
      }
      return true;
    }
    OUTCOME_TEMPLATE(class T)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<R>() != std::declval<T>()))
    constexpr bool operator!=(const success_type<T> &o) const noexcept( //
    noexcept(std::declval<R>() != std::declval<T>()))
    {
      if(this->_state._status.have_value())
      {
        return this->_state._value != o.value();
      }
      return false;
    }
    constexpr bool operator!=(const success_type<void> &o) const noexcept
    {
      (void) o;
      return !this->_state._status.have_value();
    }
    OUTCOME_TEMPLATE(class T)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<S>() != std::declval<T>()))
    constexpr bool operator!=(const failure_type<T, void> &o) const noexcept( //
    noexcept(std::declval<S>() != std::declval<T>()))
    {
      if(this->_state._status.have_error())
      {
        return this->_state._error != o.error();
      }
      return true;
    }
  };
  template <class T, class U, class V, class W> constexpr inline bool operator==(const success_type<W> &a, const basic_result_final<T, U, V> &b) noexcept(noexcept(b == a)) { return b == a; }
  template <class T, class U, class V, class W> constexpr inline bool operator==(const failure_type<W, void> &a, const basic_result_final<T, U, V> &b) noexcept(noexcept(b == a)) { return b == a; }
  template <class T, class U, class V, class W> constexpr inline bool operator!=(const success_type<W> &a, const basic_result_final<T, U, V> &b) noexcept(noexcept(b == a)) { return b != a; }
  template <class T, class U, class V, class W> constexpr inline bool operator!=(const failure_type<W, void> &a, const basic_result_final<T, U, V> &b) noexcept(noexcept(b == a)) { return b != a; }
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
/* Policies for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (13 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_ALL_NARROW_HPP
#define OUTCOME_POLICY_ALL_NARROW_HPP
/* Policies for result and outcome
(C) 2017-2020 Niall Douglas <http://www.nedproductions.biz/> (6 commits) and Andrzej Krzemieński <akrzemi1@gmail.com> (1 commit)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_BASE_HPP
#define OUTCOME_POLICY_BASE_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
namespace hooks
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_result_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_result_copy_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_result_move_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U, class... Args>
  constexpr inline void hook_result_in_place_construction(T * /*unused*/, in_place_type_t<U> /*unused*/, Args &&... /*unused*/) noexcept
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class... U> constexpr inline void hook_outcome_construction(T * /*unused*/, U &&... /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_outcome_copy_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_outcome_move_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U, class... Args>
  constexpr inline void hook_outcome_in_place_construction(T * /*unused*/, in_place_type_t<U> /*unused*/, Args &&... /*unused*/) noexcept
  {
  }
} // namespace hooks
#endif
namespace policy
{
  namespace detail
  {
    using OUTCOME_V2_NAMESPACE::detail::make_ub;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  struct base
  {
    template <class... Args> static constexpr void _silence_unused(Args &&... /*unused*/) noexcept {}
  protected:
    template <class Impl> static constexpr void _make_ub(Impl &&self) noexcept { return detail::make_ub(static_cast<Impl &&>(self)); }
    template <class Impl> static constexpr bool _has_value(Impl &&self) noexcept { return self._state._status.have_value(); }
    template <class Impl> static constexpr bool _has_error(Impl &&self) noexcept { return self._state._status.have_error(); }
    template <class Impl> static constexpr bool _has_exception(Impl &&self) noexcept { return self._state._status.have_exception(); }
    template <class Impl> static constexpr bool _has_error_is_errno(Impl &&self) noexcept { return self._state._status.have_error_is_errno(); }
    template <class Impl> static constexpr void _set_has_value(Impl &&self, bool v) noexcept { self._state._status.set_have_value(v); }
    template <class Impl> static constexpr void _set_has_error(Impl &&self, bool v) noexcept { self._state._status.set_have_error(v); }
    template <class Impl> static constexpr void _set_has_exception(Impl &&self, bool v) noexcept { self._state._status.set_have_exception(v); }
    template <class Impl> static constexpr void _set_has_error_is_errno(Impl &&self, bool v) noexcept { self._state._status.set_have_error_is_errno(v); }
    template <class Impl> static constexpr auto &&_value(Impl &&self) noexcept { return static_cast<Impl &&>(self)._state._value; }
    template <class Impl> static constexpr auto &&_error(Impl &&self) noexcept { return static_cast<Impl &&>(self)._state._error; }
  public:
    template <class R, class S, class P, class NoValuePolicy, class Impl> static inline constexpr auto &&_exception(Impl &&self) noexcept;
    template <class T, class U> static constexpr inline void on_result_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_result_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U> static constexpr inline void on_result_copy_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_result_copy_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U> static constexpr inline void on_result_move_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_result_move_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U, class... Args>
    static constexpr inline void on_result_in_place_construction(T *inst, in_place_type_t<U> _, Args &&... args) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_result_in_place_construction(inst, _, static_cast<Args &&>(args)...);
#else
      (void) inst;
      (void) _;
      _silence_unused(static_cast<Args &&>(args)...);
#endif
    }
    template <class T, class... U> static constexpr inline void on_outcome_construction(T *inst, U &&... args) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_outcome_construction(inst, static_cast<U &&>(args)...);
#else
      (void) inst;
      _silence_unused(static_cast<U &&>(args)...);
#endif
    }
    template <class T, class U> static constexpr inline void on_outcome_copy_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_outcome_copy_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U> static constexpr inline void on_outcome_move_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_outcome_move_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U, class... Args>
    static constexpr inline void on_outcome_in_place_construction(T *inst, in_place_type_t<U> _, Args &&... args) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_outcome_in_place_construction(inst, _, static_cast<Args &&>(args)...);
#else
      (void) inst;
      (void) _;
      _silence_unused(static_cast<Args &&>(args)...);
#endif
    }
    template <class Impl> static constexpr void narrow_value_check(Impl &&self) noexcept
    {
      if(!_has_value(self))
      {
        _make_ub(self);
      }
    }
    template <class Impl> static constexpr void narrow_error_check(Impl &&self) noexcept
    {
      if(!_has_error(self))
      {
        _make_ub(self);
      }
    }
    template <class Impl> static constexpr void narrow_exception_check(Impl &&self) noexcept
    {
      if(!_has_exception(self))
      {
        _make_ub(self);
      }
    }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  all_narrow. Potential doc page: `all_narrow`
*/
  struct all_narrow : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self) { base::narrow_value_check(static_cast<Impl &&>(self)); }
    template <class Impl> static constexpr void wide_error_check(Impl &&self) { base::narrow_error_check(static_cast<Impl &&>(self)); }
    template <class Impl> static constexpr void wide_exception_check(Impl &&self) { base::narrow_exception_check(static_cast<Impl &&>(self)); }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
/* Policies for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (12 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_TERMINATE_HPP
#define OUTCOME_POLICY_TERMINATE_HPP
#include <cstdlib>
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  terminate. Potential doc page: `terminate`
*/
  struct terminate : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self)
    {
      if(!base::_has_value(static_cast<Impl &&>(self)))
      {
        std::abort();
      }
    }
    template <class Impl> static constexpr void wide_error_check(Impl &&self) noexcept
    {
      if(!base::_has_error(static_cast<Impl &&>(self)))
      {
        std::abort();
      }
    }
    template <class Impl> static constexpr void wide_exception_check(Impl &&self)
    {
      if(!base::_has_exception(static_cast<Impl &&>(self)))
      {
        std::abort();
      }
    }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation" // Standardese markup confuses clang
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
template <class R, class S, class NoValuePolicy> //
class basic_result;
namespace detail
{
  // These are reused by basic_outcome to save load on the compiler
  template <class value_type, class error_type> struct result_predicates
  {
    // Predicate for the implicit constructors to be available. Weakened to allow result<int, C enum>.
    static constexpr bool implicit_constructors_enabled = //
    !(trait::is_error_type<std::decay_t<value_type>>::value &&
      trait::is_error_type<std::decay_t<error_type>>::value) // both value and error types are not whitelisted error types
    && ((!detail::is_implicitly_constructible<value_type, error_type> &&
         !detail::is_implicitly_constructible<error_type, value_type>) // if value and error types cannot be constructed into one another
        || (trait::is_error_type<std::decay_t<error_type>>::value // if error type is a whitelisted error type
            && !detail::is_implicitly_constructible<error_type, value_type> // AND which cannot be constructed from the value type
            && std::is_integral<value_type>::value)); // AND the value type is some integral type
    // Predicate for the value converting constructor to be available. Weakened to allow result<int, C enum>.
    template <class T>
    static constexpr bool enable_value_converting_constructor = //
    implicit_constructors_enabled //
    && !is_in_place_type_t<std::decay_t<T>>::value // not in place construction
    && !trait::is_error_type_enum<error_type, std::decay_t<T>>::value // not an enum valid for my error type
    && ((detail::is_implicitly_constructible<value_type, T> && !detail::is_implicitly_constructible<error_type, T>) // is unambiguously for value type
        || (std::is_same<value_type, std::decay_t<T>>::value // OR is my value type exactly
            && detail::is_implicitly_constructible<value_type, T>) ); // and my value type is constructible from this ref form of T
    // Predicate for the error converting constructor to be available. Weakened to allow result<int, C enum>.
    template <class T>
    static constexpr bool enable_error_converting_constructor = //
    implicit_constructors_enabled //
    && !is_in_place_type_t<std::decay_t<T>>::value // not in place construction
    && !trait::is_error_type_enum<error_type, std::decay_t<T>>::value // not an enum valid for my error type
    && ((!detail::is_implicitly_constructible<value_type, T> && detail::is_implicitly_constructible<error_type, T>) // is unambiguously for error type
        || (std::is_same<error_type, std::decay_t<T>>::value // OR is my error type exactly
            && detail::is_implicitly_constructible<error_type, T>) ); // and my error type is constructible from this ref form of T
    // Predicate for the error condition converting constructor to be available.
    template <class ErrorCondEnum>
    static constexpr bool enable_error_condition_converting_constructor = //
    !is_in_place_type_t<std::decay_t<ErrorCondEnum>>::value // not in place construction
    && trait::is_error_type_enum<error_type, std::decay_t<ErrorCondEnum>>::value // is an error condition enum
    /*&& !detail::is_implicitly_constructible<value_type, ErrorCondEnum> && !detail::is_implicitly_constructible<error_type, ErrorCondEnum>*/; // not
                                                                                                                                                // constructible
                                                                                                                                                // via any other
                                                                                                                                                // means
    // Predicate for the converting constructor from a compatible input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_compatible_conversion = //
    (std::is_void<T>::value ||
     detail::is_explicitly_constructible<value_type, typename basic_result<T, U, V>::value_type>) // if our value types are constructible
    &&(std::is_void<U>::value ||
       detail::is_explicitly_constructible<error_type, typename basic_result<T, U, V>::error_type>) // if our error types are constructible
    ;
    // Predicate for the converting constructor from a make_error_code() of the input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_make_error_code_compatible_conversion = //
    trait::is_error_code_available<std::decay_t<error_type>>::value // if error type has an error code
    && !enable_compatible_conversion<T, U, V> // and the normal compatible conversion is not available
    && (std::is_void<T>::value ||
        detail::is_explicitly_constructible<value_type, typename basic_result<T, U, V>::value_type>) // and if our value types are constructible
    &&detail::is_explicitly_constructible<error_type,
                                          typename trait::is_error_code_available<U>::type>; // and our error type is constructible from a make_error_code()
    // Predicate for the converting constructor from a make_exception_ptr() of the input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_make_exception_ptr_compatible_conversion = //
    trait::is_exception_ptr_available<std::decay_t<error_type>>::value // if error type has an exception ptr
    && !enable_compatible_conversion<T, U, V> // and the normal compatible conversion is not available
    && (std::is_void<T>::value ||
        detail::is_explicitly_constructible<value_type, typename basic_result<T, U, V>::value_type>) // and if our value types are constructible
    &&detail::is_explicitly_constructible<error_type, typename trait::is_exception_ptr_available<U>::type>; // and our error type is constructible from a
                                                                                                             // make_exception_ptr()
    // Predicate for the implicit converting inplace constructor from a compatible input to be available.
    struct disable_inplace_value_error_constructor;
    template <class... Args>
    using choose_inplace_value_error_constructor = std::conditional_t< //
    detail::is_constructible<value_type, Args...> && detail::is_constructible<error_type, Args...>, //
    disable_inplace_value_error_constructor, //
    std::conditional_t< //
    detail::is_constructible<value_type, Args...>, //
    value_type, //
    std::conditional_t< //
    detail::is_constructible<error_type, Args...>, //
    error_type, //
    disable_inplace_value_error_constructor>>>;
    template <class... Args>
    static constexpr bool enable_inplace_value_error_constructor =
    implicit_constructors_enabled //
    && !std::is_same<choose_inplace_value_error_constructor<Args...>, disable_inplace_value_error_constructor>::value;
  };
  template <class T, class U> constexpr inline const U &extract_value_from_success(const success_type<U> &v) { return v.value(); }
  template <class T, class U> constexpr inline U &&extract_value_from_success(success_type<U> &&v) { return static_cast<success_type<U> &&>(v).value(); }
  template <class T> constexpr inline T extract_value_from_success(const success_type<void> & /*unused*/) { return T{}; }
  template <class T, class U, class V> constexpr inline const U &extract_error_from_failure(const failure_type<U, V> &v) { return v.error(); }
  template <class T, class U, class V> constexpr inline U &&extract_error_from_failure(failure_type<U, V> &&v)
  {
    return static_cast<failure_type<U, V> &&>(v).error();
  }
  template <class T, class V> constexpr inline T extract_error_from_failure(const failure_type<void, V> & /*unused*/) { return T{}; }
  template <class T> struct is_basic_result
  {
    static constexpr bool value = false;
  };
  template <class R, class S, class T> struct is_basic_result<basic_result<R, S, T>>
  {
    static constexpr bool value = true;
  };
} // namespace detail
/*! AWAITING HUGO JSON CONVERSION TOOL
type alias template <class T> is_basic_result. Potential doc page: `is_basic_result<T>`
*/
template <class T> using is_basic_result = detail::is_basic_result<std::decay_t<T>>;
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> static constexpr bool is_basic_result_v = detail::is_basic_result<std::decay_t<T>>::value;
namespace concepts
{
#if defined(__cpp_concepts)
  /* The `basic_result` concept.
  \requires That `U` matches a `basic_result`.
  */
  template <class U>
  concept OUTCOME_GCC6_CONCEPT_BOOL basic_result =
  OUTCOME_V2_NAMESPACE::is_basic_result<U>::value ||
  (requires(U v) { OUTCOME_V2_NAMESPACE::basic_result<typename U::value_type, typename U::error_type, typename U::no_value_policy_type>(v); } && //
   detail::convertible<U, OUTCOME_V2_NAMESPACE::basic_result<typename U::value_type, typename U::error_type, typename U::no_value_policy_type>> && //
   detail::base_of<OUTCOME_V2_NAMESPACE::basic_result<typename U::value_type, typename U::error_type, typename U::no_value_policy_type>, U>);
#else
  namespace detail
  {
    inline no_match match_basic_result(...);
    template <class R, class S, class NVP, class T, //
              typename = typename T::value_type, //
              typename = typename T::error_type, //
              typename = typename T::no_value_policy_type, //
              typename std::enable_if_t<std::is_convertible<T, OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP>>::value && //
                                        std::is_base_of<OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP>, T>::value,
                                        bool> = true>
    inline OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP> match_basic_result(OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP> &&, T &&);
    template <class U>
    static constexpr bool basic_result = OUTCOME_V2_NAMESPACE::is_basic_result<U>::value ||
                                         !std::is_same<no_match, decltype(match_basic_result(std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>(),
                                                                                             std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
  } // namespace detail
  /* The `basic_result` concept.
  \requires That `U` matches a `basic_result`.
  */
  template <class U> static constexpr bool basic_result = detail::basic_result<U>;
#endif
} // namespace concepts
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
namespace hooks
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S, class NoValuePolicy> constexpr inline uint16_t spare_storage(const detail::basic_result_storage<R, S, NoValuePolicy> *r) noexcept
  {
    return r->_state._status.spare_storage_value;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S, class NoValuePolicy>
  constexpr inline void set_spare_storage(detail::basic_result_storage<R, S, NoValuePolicy> *r, uint16_t v) noexcept
  {
    r->_state._status.spare_storage_value = v;
  }
} // namespace hooks
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class R, class S, class NoValuePolicy> basic_result. Potential doc page: `basic_result<T, E, NoValuePolicy>`
*/
template <class R, class S, class NoValuePolicy> //
class OUTCOME_NODISCARD basic_result : public detail::basic_result_final<R, S, NoValuePolicy>
{
  static_assert(trait::type_can_be_used_in_basic_result<R>, "The type R cannot be used in a basic_result");
  static_assert(trait::type_can_be_used_in_basic_result<S>, "The type S cannot be used in a basic_result");
  using base = detail::basic_result_final<R, S, NoValuePolicy>;
  struct implicit_constructors_disabled_tag
  {
  };
  struct value_converting_constructor_tag
  {
  };
  struct error_converting_constructor_tag
  {
  };
  struct error_condition_converting_constructor_tag
  {
  };
  struct explicit_valueornone_converting_constructor_tag
  {
  };
  struct explicit_valueorerror_converting_constructor_tag
  {
  };
  struct explicit_compatible_copy_conversion_tag
  {
  };
  struct explicit_compatible_move_conversion_tag
  {
  };
  struct explicit_make_error_code_compatible_copy_conversion_tag
  {
  };
  struct explicit_make_error_code_compatible_move_conversion_tag
  {
  };
  struct explicit_make_exception_ptr_compatible_copy_conversion_tag
  {
  };
  struct explicit_make_exception_ptr_compatible_move_conversion_tag
  {
  };
public:
  using value_type = R;
  using error_type = S;
  using no_value_policy_type = NoValuePolicy;
  using value_type_if_enabled = typename base::_value_type;
  using error_type_if_enabled = typename base::_error_type;
  template <class T, class U = S, class V = NoValuePolicy> using rebind = basic_result<T, U, V>;
public:
  // Requirement predicates for result.
  struct predicate
  {
    using base = detail::result_predicates<value_type, error_type>;
    // Predicate for any constructors to be available at all
    static constexpr bool constructors_enabled = !std::is_same<std::decay_t<value_type>, std::decay_t<error_type>>::value;
    // Predicate for implicit constructors to be available at all
    static constexpr bool implicit_constructors_enabled = constructors_enabled && base::implicit_constructors_enabled;
    // Predicate for the value converting constructor to be available.
    template <class T>
    static constexpr bool enable_value_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_result>::value // not my type
    && base::template enable_value_converting_constructor<T>;
    // Predicate for the error converting constructor to be available.
    template <class T>
    static constexpr bool enable_error_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_result>::value // not my type
    && base::template enable_error_converting_constructor<T>;
    // Predicate for the error condition converting constructor to be available.
    template <class ErrorCondEnum>
    static constexpr bool enable_error_condition_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<ErrorCondEnum>, basic_result>::value // not my type
    && base::template enable_error_condition_converting_constructor<ErrorCondEnum>;
    // Predicate for the converting constructor from a compatible input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_result<T, U, V>, basic_result>::value // not my type
    && base::template enable_compatible_conversion<T, U, V>;
    // Predicate for the converting constructor from a make_error_code() of the input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_make_error_code_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_result<T, U, V>, basic_result>::value // not my type
    && base::template enable_make_error_code_compatible_conversion<T, U, V>;
    // Predicate for the converting constructor from a make_exception_ptr() of the input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_make_exception_ptr_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_result<T, U, V>, basic_result>::value // not my type
    && base::template enable_make_exception_ptr_compatible_conversion<T, U, V>;
    // Predicate for the inplace construction of value to be available.
    template <class... Args>
    static constexpr bool enable_inplace_value_constructor = //
    constructors_enabled //
    && (std::is_void<value_type>::value //
        || detail::is_constructible<value_type, Args...>);
    // Predicate for the inplace construction of error to be available.
    template <class... Args>
    static constexpr bool enable_inplace_error_constructor = //
    constructors_enabled //
    && (std::is_void<error_type>::value //
        || detail::is_constructible<error_type, Args...>);
    // Predicate for the implicit converting inplace constructor to be available.
    template <class... Args>
    static constexpr bool enable_inplace_value_error_constructor = //
    constructors_enabled //
    &&base::template enable_inplace_value_error_constructor<Args...>;
    template <class... Args> using choose_inplace_value_error_constructor = typename base::template choose_inplace_value_error_constructor<Args...>;
  };
public:
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result() = delete;
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result(basic_result && /*unused*/) = default; // NOLINT
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result(const basic_result & /*unused*/) = default;
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result &operator=(basic_result && /*unused*/) = default; // NOLINT
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result &operator=(const basic_result & /*unused*/) = default;
  ~basic_result() = default;
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class Arg, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!predicate::constructors_enabled && (sizeof...(Args) >= 0)))
  basic_result(Arg && /*unused*/, Args &&... /*unused*/) = delete; // NOLINT basic_result<T, T> is NOT SUPPORTED, see docs!
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED((predicate::constructors_enabled && !predicate::implicit_constructors_enabled //
                                   && (detail::is_implicitly_constructible<value_type, T> || detail::is_implicitly_constructible<error_type, T>) )))
  basic_result(T && /*unused*/, implicit_constructors_disabled_tag /*unused*/ = implicit_constructors_disabled_tag()) =
  delete; // NOLINT Implicit constructors disabled, use explicit in_place_type<T>, success() or failure(). see docs!
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_value_converting_constructor<T>))
  constexpr basic_result(T &&t, value_converting_constructor_tag /*unused*/ = value_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<typename base::value_type>, static_cast<T &&>(t)}
  {
    no_value_policy_type::on_result_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_error_converting_constructor<T>))
  constexpr basic_result(T &&t, error_converting_constructor_tag /*unused*/ = error_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<typename base::error_type>, static_cast<T &&>(t)}
  {
    no_value_policy_type::on_result_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class ErrorCondEnum)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(error_type(make_error_code(ErrorCondEnum()))), //
                    OUTCOME_TPRED(predicate::template enable_error_condition_converting_constructor<ErrorCondEnum>))
  constexpr basic_result(ErrorCondEnum &&t, error_condition_converting_constructor_tag /*unused*/ = error_condition_converting_constructor_tag()) noexcept(
  noexcept(error_type(make_error_code(static_cast<ErrorCondEnum &&>(t))))) // NOLINT
      : base{in_place_type<typename base::error_type>, make_error_code(t)}
  {
    no_value_policy_type::on_result_construction(this, static_cast<ErrorCondEnum &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(convert::value_or_error<basic_result, std::decay_t<T>>::enable_result_inputs || !concepts::basic_result<T>), //
                    OUTCOME_TEXPR(convert::value_or_error<basic_result, std::decay_t<T>>{}(std::declval<T>())))
  constexpr explicit basic_result(T &&o,
                                  explicit_valueorerror_converting_constructor_tag /*unused*/ = explicit_valueorerror_converting_constructor_tag()) // NOLINT
      : basic_result{convert::value_or_error<basic_result, std::decay_t<T>>{}(static_cast<T &&>(o))}
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(
  const basic_result<T, U, V> &o,
  explicit_compatible_copy_conversion_tag /*unused*/ =
  explicit_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>)
      : base{typename base::compatible_conversion_tag(), o}
  {
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(
  basic_result<T, U, V> &&o,
  explicit_compatible_move_conversion_tag /*unused*/ =
  explicit_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>)
      : base{typename base::compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
  {
    no_value_policy_type::on_result_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(const basic_result<T, U, V> &o,
                                  explicit_make_error_code_compatible_copy_conversion_tag /*unused*/ =
                                  explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                      &&noexcept(make_error_code(std::declval<U>())))
      : base{typename base::make_error_code_compatible_conversion_tag(), o}
  {
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(basic_result<T, U, V> &&o,
                                  explicit_make_error_code_compatible_move_conversion_tag /*unused*/ =
                                  explicit_make_error_code_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                      &&noexcept(make_error_code(std::declval<U>())))
      : base{typename base::make_error_code_compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
  {
    no_value_policy_type::on_result_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(const basic_result<T, U, V> &o,
                                  explicit_make_exception_ptr_compatible_copy_conversion_tag /*unused*/ =
                                  explicit_make_exception_ptr_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                         &&noexcept(make_exception_ptr(std::declval<U>())))
      : base{typename base::make_exception_ptr_compatible_conversion_tag(), o}
  {
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(basic_result<T, U, V> &&o,
                                  explicit_make_exception_ptr_compatible_move_conversion_tag /*unused*/ =
                                  explicit_make_exception_ptr_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                         &&noexcept(make_exception_ptr(std::declval<U>())))
      : base{typename base::make_exception_ptr_compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
  {
    no_value_policy_type::on_result_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<Args...>))
  constexpr explicit basic_result(in_place_type_t<value_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, Args...>)
      : base{_, static_cast<Args &&>(args)...}
  {
    no_value_policy_type::on_result_in_place_construction(this, in_place_type<value_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_result(in_place_type_t<value_type_if_enabled> _, std::initializer_list<U> il,
                                  Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, std::initializer_list<U>, Args...>)
      : base{_, il, static_cast<Args &&>(args)...}
  {
    no_value_policy_type::on_result_in_place_construction(this, in_place_type<value_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<Args...>))
  constexpr explicit basic_result(in_place_type_t<error_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, Args...>)
      : base{_, static_cast<Args &&>(args)...}
  {
    no_value_policy_type::on_result_in_place_construction(this, in_place_type<error_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_result(in_place_type_t<error_type_if_enabled> _, std::initializer_list<U> il,
                                  Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, std::initializer_list<U>, Args...>)
      : base{_, il, static_cast<Args &&>(args)...}
  {
    no_value_policy_type::on_result_in_place_construction(this, in_place_type<error_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class A1, class A2, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_error_constructor<A1, A2, Args...>))
  constexpr basic_result(A1 &&a1, A2 &&a2, Args &&... args) noexcept(noexcept(
  typename predicate::template choose_inplace_value_error_constructor<A1, A2, Args...>(std::declval<A1>(), std::declval<A2>(), std::declval<Args>()...)))
      : basic_result(in_place_type<typename predicate::template choose_inplace_value_error_constructor<A1, A2, Args...>>, static_cast<A1 &&>(a1),
                     static_cast<A2 &&>(a2), static_cast<Args &&>(args)...)
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  constexpr basic_result(const success_type<void> &o) noexcept(std::is_nothrow_default_constructible<value_type>::value) // NOLINT
      : base{in_place_type<value_type_if_enabled>}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, void, void>))
  constexpr basic_result(const success_type<T> &o) noexcept(detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<value_type_if_enabled>, detail::extract_value_from_success<value_type>(o)}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<T, void, void>))
  constexpr basic_result(success_type<T> &&o) noexcept(detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<value_type_if_enabled>, detail::extract_value_from_success<value_type>(static_cast<success_type<T> &&>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_move_construction(this, static_cast<success_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<void, T, void>))
  constexpr basic_result(const failure_type<T> &o, explicit_compatible_copy_conversion_tag /*unused*/ = explicit_compatible_copy_conversion_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<error_type_if_enabled>, detail::extract_error_from_failure<error_type>(o)}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<void, T, void>))
  constexpr basic_result(failure_type<T> &&o, explicit_compatible_move_conversion_tag /*unused*/ = explicit_compatible_move_conversion_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<error_type_if_enabled>, detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_move_construction(this, static_cast<failure_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<void, T, void>))
  constexpr basic_result(const failure_type<T> &o,
                         explicit_make_error_code_compatible_copy_conversion_tag /*unused*/ =
                         explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>()))) // NOLINT
      : base{in_place_type<error_type_if_enabled>, make_error_code(detail::extract_error_from_failure<error_type>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<void, T, void>))
  constexpr basic_result(failure_type<T> &&o,
                         explicit_make_error_code_compatible_move_conversion_tag /*unused*/ =
                         explicit_make_error_code_compatible_move_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>()))) // NOLINT
      : base{in_place_type<error_type_if_enabled>, make_error_code(detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o)))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_move_construction(this, static_cast<failure_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<void, T, void>))
  constexpr basic_result(const failure_type<T> &o,
                         explicit_make_exception_ptr_compatible_copy_conversion_tag /*unused*/ =
                         explicit_make_exception_ptr_compatible_copy_conversion_tag()) noexcept(noexcept(make_exception_ptr(std::declval<T>()))) // NOLINT
      : base{in_place_type<error_type_if_enabled>, make_exception_ptr(detail::extract_error_from_failure<error_type>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<void, T, void>))
  constexpr basic_result(failure_type<T> &&o,
                         explicit_make_exception_ptr_compatible_move_conversion_tag /*unused*/ =
                         explicit_make_exception_ptr_compatible_move_conversion_tag()) noexcept(noexcept(make_exception_ptr(std::declval<T>()))) // NOLINT
      : base{in_place_type<error_type_if_enabled>, make_exception_ptr(detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o)))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_move_construction(this, static_cast<failure_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  constexpr void swap(basic_result &o) noexcept((std::is_void<value_type>::value || detail::is_nothrow_swappable<value_type>::value) //
                                                && (std::is_void<error_type>::value || detail::is_nothrow_swappable<error_type>::value))
  {
    this->_state.swap(o._state);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  auto as_failure() const & { return failure(this->assume_error(), hooks::spare_storage(this)); }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  auto as_failure() &&
  {
    this->_state._status.set_have_moved_from(true);
    return failure(static_cast<basic_result &&>(*this).assume_error(), hooks::spare_storage(this));
  }
#ifdef __APPLE__
  failure_type<error_type> _xcode_workaround_as_failure() &&;
#endif
};
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S, class P> inline void swap(basic_result<R, S, P> &a, basic_result<R, S, P> &b) noexcept(noexcept(a.swap(b)))
{
  a.swap(b);
}
#if !defined(NDEBUG)
// Check is trivial in all ways except default constructibility
// static_assert(std::is_trivial<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivial!");
// static_assert(std::is_trivially_default_constructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially default
// constructible!");
static_assert(std::is_trivially_copyable<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially copyable!");
static_assert(std::is_trivially_assignable<basic_result<int, long, policy::all_narrow>, basic_result<int, long, policy::all_narrow>>::value,
              "result<int> is not trivially assignable!");
static_assert(std::is_trivially_destructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially destructible!");
static_assert(std::is_trivially_copy_constructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially copy constructible!");
static_assert(std::is_trivially_move_constructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially move constructible!");
static_assert(std::is_trivially_copy_assignable<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially copy assignable!");
static_assert(std::is_trivially_move_assignable<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially move assignable!");
// Also check is standard layout
static_assert(std::is_standard_layout<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not a standard layout type!");
#endif
OUTCOME_V2_NAMESPACE_END
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif
/* Exception observers for outcome type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (3 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_OUTCOME_EXCEPTION_OBSERVERS_HPP
#define OUTCOME_BASIC_OUTCOME_EXCEPTION_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class Base, class R, class S, class P, class NoValuePolicy> class basic_outcome_exception_observers : public Base
  {
  public:
    using exception_type = P;
    using Base::Base;
    constexpr inline exception_type &assume_exception() & noexcept;
    constexpr inline const exception_type &assume_exception() const &noexcept;
    constexpr inline exception_type &&assume_exception() && noexcept;
    constexpr inline const exception_type &&assume_exception() const &&noexcept;
    constexpr inline exception_type &exception() &;
    constexpr inline const exception_type &exception() const &;
    constexpr inline exception_type &&exception() &&;
    constexpr inline const exception_type &&exception() const &&;
  };
  // Exception observers not present
  template <class Base, class R, class S, class NoValuePolicy> class basic_outcome_exception_observers<Base, R, S, void, NoValuePolicy> : public Base
  {
  public:
    using Base::Base;
    constexpr void assume_exception() const noexcept { NoValuePolicy::narrow_exception_check(this); }
    constexpr void exception() const { NoValuePolicy::wide_exception_check(this); }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
/* Failure observers for outcome type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (7 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_OUTCOME_FAILURE_OBSERVERS_HPP
#define OUTCOME_BASIC_OUTCOME_FAILURE_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  namespace adl
  {
    struct search_detail_adl
    {
    };
    // Do NOT use template requirements here!
    template <class S, typename = decltype(basic_outcome_failure_exception_from_error(std::declval<S>()))>
    inline auto _delayed_lookup_basic_outcome_failure_exception_from_error(const S &ec, search_detail_adl /*unused*/)
    {
      // ADL discovered
      return basic_outcome_failure_exception_from_error(ec);
    }
  } // namespace adl
#if defined(_MSC_VER) && _MSC_VER <= 1923 // VS2019
  // VS2017 and VS2019 with /permissive- chokes on the correct form due to over eager early instantiation.
  template <class S, class P> inline void _delayed_lookup_basic_outcome_failure_exception_from_error(...) { static_assert(sizeof(S) == 0, "No specialisation for these error and exception types available!"); }
#else
  template <class S, class P> inline void _delayed_lookup_basic_outcome_failure_exception_from_error(...) = delete; // NOLINT No specialisation for these error and exception types available!
#endif
  template <class exception_type> inline exception_type current_exception_or_fatal(std::exception_ptr e) { std::rethrow_exception(e); }
  template <> inline std::exception_ptr current_exception_or_fatal<std::exception_ptr>(std::exception_ptr e) { return e; }
  template <class Base, class R, class S, class P, class NoValuePolicy> class basic_outcome_failure_observers : public Base
  {
  public:
    using exception_type = P;
    using Base::Base;
    exception_type failure() const noexcept
    {
#ifdef __cpp_exceptions
      try
#endif
      {
        if(this->_state._status.have_exception())
        {
          return this->assume_exception();
        }
        if(this->_state._status.have_error())
        {
          return _delayed_lookup_basic_outcome_failure_exception_from_error(this->assume_error(), adl::search_detail_adl());
        }
        return exception_type();
      }
#ifdef __cpp_exceptions
      catch(...)
      {
        // Return the failure if exception_type is std::exception_ptr,
        // otherwise terminate same as throwing an exception inside noexcept
        return current_exception_or_fatal<exception_type>(std::current_exception());
      }
#endif
    }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation" // Standardese markup confuses clang
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
template <class R, class S, class P, class NoValuePolicy> //
class basic_outcome;
namespace detail
{
  // May be reused by basic_outcome subclasses to save load on the compiler
  template <class value_type, class error_type, class exception_type> struct outcome_predicates
  {
    using result = result_predicates<value_type, error_type>;
    // Predicate for the implicit constructors to be available
    static constexpr bool implicit_constructors_enabled = //
    result::implicit_constructors_enabled //
    && !detail::is_implicitly_constructible<value_type, exception_type> //
    && !detail::is_implicitly_constructible<error_type, exception_type> //
    && !detail::is_implicitly_constructible<exception_type, value_type> //
    && !detail::is_implicitly_constructible<exception_type, error_type>;
    // Predicate for the value converting constructor to be available.
    template <class T>
    static constexpr bool enable_value_converting_constructor = //
    implicit_constructors_enabled //
    &&result::template enable_value_converting_constructor<T> //
    && !detail::is_implicitly_constructible<exception_type, T>; // deliberately less tolerant of ambiguity than result's edition
    // Predicate for the error converting constructor to be available.
    template <class T>
    static constexpr bool enable_error_converting_constructor = //
    implicit_constructors_enabled //
    &&result::template enable_error_converting_constructor<T> //
    && !detail::is_implicitly_constructible<exception_type, T>; // deliberately less tolerant of ambiguity than result's edition
    // Predicate for the error condition converting constructor to be available.
    template <class ErrorCondEnum>
    static constexpr bool enable_error_condition_converting_constructor = result::template enable_error_condition_converting_constructor<ErrorCondEnum> //
                                                                          && !detail::is_implicitly_constructible<exception_type, ErrorCondEnum>;
    // Predicate for the exception converting constructor to be available.
    template <class T>
    static constexpr bool enable_exception_converting_constructor = //
    implicit_constructors_enabled //
    && !is_in_place_type_t<std::decay_t<T>>::value // not in place construction
    && !detail::is_implicitly_constructible<value_type, T> && !detail::is_implicitly_constructible<error_type, T> &&
    detail::is_implicitly_constructible<exception_type, T>;
    // Predicate for the error + exception converting constructor to be available.
    template <class T, class U>
    static constexpr bool enable_error_exception_converting_constructor = //
    implicit_constructors_enabled //
    && !is_in_place_type_t<std::decay_t<T>>::value // not in place construction
    && !detail::is_implicitly_constructible<value_type, T> && detail::is_implicitly_constructible<error_type, T> //
    && !detail::is_implicitly_constructible<value_type, U> && detail::is_implicitly_constructible<exception_type, U>;
    // Predicate for the converting copy constructor from a compatible outcome to be available.
    template <class T, class U, class V, class W>
    static constexpr bool enable_compatible_conversion = //
    (std::is_void<T>::value ||
     detail::is_explicitly_constructible<value_type, typename basic_outcome<T, U, V, W>::value_type>) // if our value types are constructible
    &&(std::is_void<U>::value ||
       detail::is_explicitly_constructible<error_type, typename basic_outcome<T, U, V, W>::error_type>) // if our error types are constructible
    &&(std::is_void<V>::value ||
       detail::is_explicitly_constructible<exception_type, typename basic_outcome<T, U, V, W>::exception_type>) // if our exception types are constructible
    ;
    // Predicate for the converting constructor from a make_error_code() of the input to be available.
    template <class T, class U, class V, class W>
    static constexpr bool enable_make_error_code_compatible_conversion = //
    trait::is_error_code_available<std::decay_t<error_type>>::value // if error type has an error code
    && !enable_compatible_conversion<T, U, V, W> // and the normal compatible conversion is not available
    && (std::is_void<T>::value ||
        detail::is_explicitly_constructible<value_type, typename basic_outcome<T, U, V, W>::value_type>) // and if our value types are constructible
    &&detail::is_explicitly_constructible<error_type,
                                          typename trait::is_error_code_available<U>::type> // and our error type is constructible from a make_error_code()
    && (std::is_void<V>::value ||
        detail::is_explicitly_constructible<exception_type, typename basic_outcome<T, U, V, W>::exception_type>); // and our exception types are constructible
    // Predicate for the implicit converting inplace constructor from a compatible input to be available.
    struct disable_inplace_value_error_exception_constructor;
    template <class... Args>
    using choose_inplace_value_error_exception_constructor = std::conditional_t< //
    ((static_cast<int>(detail::is_constructible<value_type, Args...>) + static_cast<int>(detail::is_constructible<error_type, Args...>) +
      static_cast<int>(detail::is_constructible<exception_type, Args...>)) > 1), //
    disable_inplace_value_error_exception_constructor, //
    std::conditional_t< //
    detail::is_constructible<value_type, Args...>, //
    value_type, //
    std::conditional_t< //
    detail::is_constructible<error_type, Args...>, //
    error_type, //
    std::conditional_t< //
    detail::is_constructible<exception_type, Args...>, //
    exception_type, //
    disable_inplace_value_error_exception_constructor>>>>;
    template <class... Args>
    static constexpr bool enable_inplace_value_error_exception_constructor = //
    implicit_constructors_enabled &&
    !std::is_same<choose_inplace_value_error_exception_constructor<Args...>, disable_inplace_value_error_exception_constructor>::value;
  };
  // Select whether to use basic_outcome_failure_observers or not
  template <class Base, class R, class S, class P, class NoValuePolicy>
  using select_basic_outcome_failure_observers = //
  std::conditional_t<trait::is_error_code_available<S>::value && trait::is_exception_ptr_available<P>::value,
                     basic_outcome_failure_observers<Base, R, S, P, NoValuePolicy>, Base>;
  template <class T, class U, class V> constexpr inline const V &extract_exception_from_failure(const failure_type<U, V> &v) { return v.exception(); }
  template <class T, class U, class V> constexpr inline V &&extract_exception_from_failure(failure_type<U, V> &&v)
  {
    return static_cast<failure_type<U, V> &&>(v).exception();
  }
  template <class T, class U> constexpr inline const U &extract_exception_from_failure(const failure_type<U, void> &v) { return v.error(); }
  template <class T, class U> constexpr inline U &&extract_exception_from_failure(failure_type<U, void> &&v)
  {
    return static_cast<failure_type<U, void> &&>(v).error();
  }
  template <class T> struct is_basic_outcome
  {
    static constexpr bool value = false;
  };
  template <class R, class S, class T, class N> struct is_basic_outcome<basic_outcome<R, S, T, N>>
  {
    static constexpr bool value = true;
  };
} // namespace detail
/*! AWAITING HUGO JSON CONVERSION TOOL
type alias template <class T> is_basic_outcome. Potential doc page: `is_basic_outcome<T>`
*/
template <class T> using is_basic_outcome = detail::is_basic_outcome<std::decay_t<T>>;
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> static constexpr bool is_basic_outcome_v = detail::is_basic_outcome<std::decay_t<T>>::value;
namespace concepts
{
#if defined(__cpp_concepts)
  /* The `basic_outcome` concept.
  \requires That `U` matches a `basic_outcome`.
  */
  template <class U>
  concept OUTCOME_GCC6_CONCEPT_BOOL basic_outcome =
  OUTCOME_V2_NAMESPACE::is_basic_outcome<U>::value ||
  (requires(U v) {
    OUTCOME_V2_NAMESPACE::basic_outcome<typename U::value_type, typename U::error_type, typename U::exception_type, typename U::no_value_policy_type>(v);
  } && //
   detail::convertible<
   U, OUTCOME_V2_NAMESPACE::basic_outcome<typename U::value_type, typename U::error_type, typename U::exception_type, typename U::no_value_policy_type>> && //
   detail::base_of<
   OUTCOME_V2_NAMESPACE::basic_outcome<typename U::value_type, typename U::error_type, typename U::exception_type, typename U::no_value_policy_type>, U>);
#else
  namespace detail
  {
    inline no_match match_basic_outcome(...);
    template <class R, class S, class P, class NVP, class T, //
              typename = typename T::value_type, //
              typename = typename T::error_type, //
              typename = typename T::exception_type, //
              typename = typename T::no_value_policy_type, //
              typename std::enable_if_t<std::is_convertible<T, OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP>>::value && //
                                        std::is_base_of<OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP>, T>::value,
                                        bool> = true>
    inline OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP> match_basic_outcome(OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP> &&, T &&);
    template <class U>
    static constexpr bool basic_outcome =
    OUTCOME_V2_NAMESPACE::is_basic_outcome<U>::value ||
    !std::is_same<no_match, decltype(match_basic_outcome(std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>(),
                                                         std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
  } // namespace detail
  /* The `basic_outcome` concept.
  \requires That `U` matches a `basic_outcome`.
  */
  template <class U> static constexpr bool basic_outcome = detail::basic_outcome<U>;
#endif
} // namespace concepts
namespace hooks
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S, class P, class NoValuePolicy, class U>
  constexpr inline void override_outcome_exception(basic_outcome<R, S, P, NoValuePolicy> *o, U &&v) noexcept;
} // namespace hooks
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class R, class S, class P, class NoValuePolicy> basic_outcome. Potential doc page: `basic_outcome<T, EC, EP, NoValuePolicy>`
*/
template <class R, class S, class P, class NoValuePolicy> //
class OUTCOME_NODISCARD basic_outcome
    : public detail::select_basic_outcome_failure_observers<
      detail::basic_outcome_exception_observers<detail::basic_result_final<R, S, NoValuePolicy>, R, S, P, NoValuePolicy>, R, S, P, NoValuePolicy>
{
  static_assert(trait::type_can_be_used_in_basic_result<P>, "The exception_type cannot be used");
  static_assert(std::is_void<P>::value || std::is_default_constructible<P>::value, "exception_type must be void or default constructible");
  using base = detail::select_basic_outcome_failure_observers<
  detail::basic_outcome_exception_observers<detail::basic_result_final<R, S, NoValuePolicy>, R, S, P, NoValuePolicy>, R, S, P, NoValuePolicy>;
  friend struct policy::base;
  template <class T, class U, class V, class W> //
  friend class basic_outcome;
  template <class T, class U, class V, class W, class X>
  friend constexpr inline void hooks::override_outcome_exception(basic_outcome<T, U, V, W> *o, X &&v) noexcept; // NOLINT
  struct implicit_constructors_disabled_tag
  {
  };
  struct value_converting_constructor_tag
  {
  };
  struct error_converting_constructor_tag
  {
  };
  struct error_condition_converting_constructor_tag
  {
  };
  struct exception_converting_constructor_tag
  {
  };
  struct error_exception_converting_constructor_tag
  {
  };
  struct explicit_valueorerror_converting_constructor_tag
  {
  };
  struct explicit_compatible_copy_conversion_tag
  {
  };
  struct explicit_compatible_move_conversion_tag
  {
  };
  struct explicit_make_error_code_compatible_copy_conversion_tag
  {
  };
  struct explicit_make_error_code_compatible_move_conversion_tag
  {
  };
  struct error_failure_tag
  {
  };
  struct exception_failure_tag
  {
  };
  struct disable_in_place_value_type
  {
  };
  struct disable_in_place_error_type
  {
  };
  struct disable_in_place_exception_type
  {
  };
public:
  using value_type = R;
  using error_type = S;
  using exception_type = P;
  using no_value_policy_type = NoValuePolicy;
  template <class T, class U = S, class V = P, class W = NoValuePolicy> using rebind = basic_outcome<T, U, V, W>;
protected:
  // Requirement predicates for outcome.
  struct predicate
  {
    using base = detail::outcome_predicates<value_type, error_type, exception_type>;
    // Predicate for any constructors to be available at all
    static constexpr bool constructors_enabled =
    (!std::is_same<std::decay_t<value_type>, std::decay_t<error_type>>::value || (std::is_void<value_type>::value && std::is_void<error_type>::value)) //
    && (!std::is_same<std::decay_t<value_type>, std::decay_t<exception_type>>::value ||
        (std::is_void<value_type>::value && std::is_void<exception_type>::value)) //
    && (!std::is_same<std::decay_t<error_type>, std::decay_t<exception_type>>::value ||
        (std::is_void<error_type>::value && std::is_void<exception_type>::value)) //
    ;
    // Predicate for implicit constructors to be available at all
    static constexpr bool implicit_constructors_enabled = constructors_enabled && base::implicit_constructors_enabled;
    // Predicate for the value converting constructor to be available.
    template <class T>
    static constexpr bool enable_value_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_outcome>::value // not my type
    && base::template enable_value_converting_constructor<T>;
    // Predicate for the error converting constructor to be available.
    template <class T>
    static constexpr bool enable_error_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_outcome>::value // not my type
    && base::template enable_error_converting_constructor<T>;
    // Predicate for the error condition converting constructor to be available.
    template <class ErrorCondEnum>
    static constexpr bool enable_error_condition_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<ErrorCondEnum>, basic_outcome>::value // not my type
    && base::template enable_error_condition_converting_constructor<ErrorCondEnum>;
    // Predicate for the exception converting constructor to be available.
    template <class T>
    static constexpr bool enable_exception_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_outcome>::value // not my type
    && base::template enable_exception_converting_constructor<T>;
    // Predicate for the error + exception converting constructor to be available.
    template <class T, class U>
    static constexpr bool enable_error_exception_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_outcome>::value // not my type
    && base::template enable_error_exception_converting_constructor<T, U>;
    // Predicate for the converting constructor from a compatible input to be available.
    template <class T, class U, class V, class W>
    static constexpr bool enable_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_outcome<T, U, V, W>, basic_outcome>::value // not my type
    && base::template enable_compatible_conversion<T, U, V, W>;
    // Predicate for the converting constructor from a make_error_code() of the input to be available.
    template <class T, class U, class V, class W>
    static constexpr bool enable_make_error_code_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_outcome<T, U, V, W>, basic_outcome>::value // not my type
    && base::template enable_make_error_code_compatible_conversion<T, U, V, W>;
    // Predicate for the inplace construction of value to be available.
    template <class... Args>
    static constexpr bool enable_inplace_value_constructor = //
    constructors_enabled //
    && (std::is_void<value_type>::value //
        || detail::is_constructible<value_type, Args...>);
    // Predicate for the inplace construction of error to be available.
    template <class... Args>
    static constexpr bool enable_inplace_error_constructor = //
    constructors_enabled //
    && (std::is_void<error_type>::value //
        || detail::is_constructible<error_type, Args...>);
    // Predicate for the inplace construction of exception to be available.
    template <class... Args>
    static constexpr bool enable_inplace_exception_constructor = //
    constructors_enabled //
    && (std::is_void<exception_type>::value //
        || detail::is_constructible<exception_type, Args...>);
    // Predicate for the implicit converting inplace constructor to be available.
    template <class... Args>
    static constexpr bool enable_inplace_value_error_exception_constructor = //
    constructors_enabled //
    &&base::template enable_inplace_value_error_exception_constructor<Args...>;
    template <class... Args>
    using choose_inplace_value_error_exception_constructor = typename base::template choose_inplace_value_error_exception_constructor<Args...>;
  };
public:
  using value_type_if_enabled =
  std::conditional_t<std::is_same<value_type, error_type>::value || std::is_same<value_type, exception_type>::value, disable_in_place_value_type, value_type>;
  using error_type_if_enabled =
  std::conditional_t<std::is_same<error_type, value_type>::value || std::is_same<error_type, exception_type>::value, disable_in_place_error_type, error_type>;
  using exception_type_if_enabled = std::conditional_t<std::is_same<exception_type, value_type>::value || std::is_same<exception_type, error_type>::value,
                                                       disable_in_place_exception_type, exception_type>;
protected:
  detail::devoid<exception_type> _ptr;
public:
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class Arg, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED((!predicate::constructors_enabled && sizeof...(Args) >= 0)))
  basic_outcome(Arg && /*unused*/, Args &&... /*unused*/) = delete; // NOLINT basic_outcome<> with any of the same type is NOT SUPPORTED, see docs!
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED((predicate::constructors_enabled && !predicate::implicit_constructors_enabled //
                                   && (detail::is_implicitly_constructible<value_type, T> || detail::is_implicitly_constructible<error_type, T> ||
                                       detail::is_implicitly_constructible<exception_type, T>) )))
  basic_outcome(T && /*unused*/, implicit_constructors_disabled_tag /*unused*/ = implicit_constructors_disabled_tag()) =
  delete; // NOLINT Implicit constructors disabled, use explicit in_place_type<T>, success() or failure(). see docs!
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_value_converting_constructor<T>))
  constexpr basic_outcome(T &&t, value_converting_constructor_tag /*unused*/ = value_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<typename base::_value_type>, static_cast<T &&>(t)}
      , _ptr()
  {
    no_value_policy_type::on_outcome_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_error_converting_constructor<T>))
  constexpr basic_outcome(T &&t, error_converting_constructor_tag /*unused*/ = error_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<typename base::_error_type>, static_cast<T &&>(t)}
      , _ptr()
  {
    no_value_policy_type::on_outcome_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class ErrorCondEnum)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(error_type(make_error_code(ErrorCondEnum()))), //
                    OUTCOME_TPRED(predicate::template enable_error_condition_converting_constructor<ErrorCondEnum>))
  constexpr basic_outcome(ErrorCondEnum &&t, error_condition_converting_constructor_tag /*unused*/ = error_condition_converting_constructor_tag()) noexcept(
  noexcept(error_type(make_error_code(static_cast<ErrorCondEnum &&>(t))))) // NOLINT
      : base{in_place_type<typename base::_error_type>, make_error_code(t)}
  {
    no_value_policy_type::on_outcome_construction(this, static_cast<ErrorCondEnum &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_exception_converting_constructor<T>))
  constexpr basic_outcome(T &&t, exception_converting_constructor_tag /*unused*/ = exception_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<exception_type, T>) // NOLINT
      : base()
      , _ptr(static_cast<T &&>(t))
  {
    this->_state._status.set_have_exception(true);
    no_value_policy_type::on_outcome_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_error_exception_converting_constructor<T, U>))
  constexpr basic_outcome(T &&a, U &&b, error_exception_converting_constructor_tag /*unused*/ = error_exception_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T> &&detail::is_nothrow_constructible<exception_type, U>) // NOLINT
      : base{in_place_type<typename base::_error_type>, static_cast<T &&>(a)}
      , _ptr(static_cast<U &&>(b))
  {
    this->_state._status.set_have_exception(true);
    no_value_policy_type::on_outcome_construction(this, static_cast<T &&>(a), static_cast<U &&>(b));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(convert::value_or_error<basic_outcome, std::decay_t<T>>::enable_result_inputs || !concepts::basic_result<T>), //
                    OUTCOME_TPRED(convert::value_or_error<basic_outcome, std::decay_t<T>>::enable_outcome_inputs || !concepts::basic_outcome<T>), //
                    OUTCOME_TEXPR(convert::value_or_error<basic_outcome, std::decay_t<T>>{}(std::declval<T>())))
  constexpr explicit basic_outcome(T &&o,
                                   explicit_valueorerror_converting_constructor_tag /*unused*/ = explicit_valueorerror_converting_constructor_tag()) // NOLINT
      : basic_outcome{convert::value_or_error<basic_outcome, std::decay_t<T>>{}(static_cast<T &&>(o))}
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V, class W)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V, W>))
  constexpr explicit basic_outcome(
  const basic_outcome<T, U, V, W> &o,
  explicit_compatible_copy_conversion_tag /*unused*/ =
  explicit_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
                                                      &&detail::is_nothrow_constructible<exception_type, V>)
      : base{typename base::compatible_conversion_tag(), o}
      , _ptr(o._ptr)
  {
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V, class W)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V, W>))
  constexpr explicit basic_outcome(
  basic_outcome<T, U, V, W> &&o,
  explicit_compatible_move_conversion_tag /*unused*/ =
  explicit_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
                                                      &&detail::is_nothrow_constructible<exception_type, V>)
      : base{typename base::compatible_conversion_tag(), static_cast<basic_outcome<T, U, V, W> &&>(o)}
      , _ptr(static_cast<typename basic_outcome<T, U, V, W>::exception_type &&>(o._ptr))
  {
    no_value_policy_type::on_outcome_move_construction(this, static_cast<basic_outcome<T, U, V, W> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_compatible_conversion<T, U, V>))
  constexpr explicit basic_outcome(
  const basic_result<T, U, V> &o,
  explicit_compatible_copy_conversion_tag /*unused*/ =
  explicit_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
                                                      &&detail::is_nothrow_constructible<exception_type>)
      : base{typename base::compatible_conversion_tag(), o}
      , _ptr()
  {
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_compatible_conversion<T, U, V>))
  constexpr explicit basic_outcome(
  basic_result<T, U, V> &&o,
  explicit_compatible_move_conversion_tag /*unused*/ =
  explicit_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
                                                      &&detail::is_nothrow_constructible<exception_type>)
      : base{typename base::compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
      , _ptr()
  {
    no_value_policy_type::on_outcome_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_make_error_code_compatible_conversion<T, U, V>))
  constexpr explicit basic_outcome(const basic_result<T, U, V> &o,
                                   explicit_make_error_code_compatible_copy_conversion_tag /*unused*/ =
                                   explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                       &&noexcept(make_error_code(std::declval<U>())) &&
                                                                                                       detail::is_nothrow_constructible<exception_type>)
      : base{typename base::make_error_code_compatible_conversion_tag(), o}
      , _ptr()
  {
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_make_error_code_compatible_conversion<T, U, V>))
  constexpr explicit basic_outcome(basic_result<T, U, V> &&o,
                                   explicit_make_error_code_compatible_move_conversion_tag /*unused*/ =
                                   explicit_make_error_code_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                       &&noexcept(make_error_code(std::declval<U>())) &&
                                                                                                       detail::is_nothrow_constructible<exception_type>)
      : base{typename base::make_error_code_compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
      , _ptr()
  {
    no_value_policy_type::on_outcome_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<Args...>))
  constexpr explicit basic_outcome(in_place_type_t<value_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, Args...>)
      : base{_, static_cast<Args &&>(args)...}
      , _ptr()
  {
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<value_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_outcome(in_place_type_t<value_type_if_enabled> _, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, std::initializer_list<U>, Args...>)
      : base{_, il, static_cast<Args &&>(args)...}
      , _ptr()
  {
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<value_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<Args...>))
  constexpr explicit basic_outcome(in_place_type_t<error_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, Args...>)
      : base{_, static_cast<Args &&>(args)...}
      , _ptr()
  {
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<error_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_outcome(in_place_type_t<error_type_if_enabled> _, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, std::initializer_list<U>, Args...>)
      : base{_, il, static_cast<Args &&>(args)...}
      , _ptr()
  {
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<error_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_exception_constructor<Args...>))
  constexpr explicit basic_outcome(in_place_type_t<exception_type_if_enabled> /*unused*/,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<exception_type, Args...>)
      : base()
      , _ptr(static_cast<Args &&>(args)...)
  {
    this->_state._status.set_have_exception(true);
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<exception_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_exception_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_outcome(in_place_type_t<exception_type_if_enabled> /*unused*/, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<exception_type, std::initializer_list<U>, Args...>)
      : base()
      , _ptr(il, static_cast<Args &&>(args)...)
  {
    this->_state._status.set_have_exception(true);
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<exception_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class A1, class A2, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_error_exception_constructor<A1, A2, Args...>))
  constexpr basic_outcome(A1 &&a1, A2 &&a2, Args &&... args) noexcept(
  noexcept(typename predicate::template choose_inplace_value_error_exception_constructor<A1, A2, Args...>(std::declval<A1>(), std::declval<A2>(),
                                                                                                          std::declval<Args>()...)))
      : basic_outcome(in_place_type<typename predicate::template choose_inplace_value_error_exception_constructor<A1, A2, Args...>>, static_cast<A1 &&>(a1),
                      static_cast<A2 &&>(a2), static_cast<Args &&>(args)...)
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  constexpr basic_outcome(const success_type<void> &o) noexcept(std::is_nothrow_default_constructible<value_type>::value) // NOLINT
      : base{in_place_type<typename base::_value_type>}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<T, void, void, void>))
  constexpr basic_outcome(const success_type<T> &o) noexcept(detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<typename base::_value_type>, detail::extract_value_from_success<value_type>(o)}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<T, void, void, void>))
  constexpr basic_outcome(success_type<T> &&o) noexcept(detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<typename base::_value_type>, detail::extract_value_from_success<value_type>(static_cast<success_type<T> &&>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_move_construction(this, static_cast<success_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, T, void, void>))
  constexpr basic_outcome(const failure_type<T> &o,
                          error_failure_tag /*unused*/ = error_failure_tag()) noexcept(detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(o)}
      , _ptr()
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, void, T, void>))
  constexpr basic_outcome(const failure_type<T> &o,
                          exception_failure_tag /*unused*/ = exception_failure_tag()) noexcept(detail::is_nothrow_constructible<exception_type, T>) // NOLINT
      : base()
      , _ptr(detail::extract_exception_from_failure<exception_type>(o))
  {
    this->_state._status.set_have_exception(true);
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_make_error_code_compatible_conversion<void, T, void, void>))
  constexpr basic_outcome(const failure_type<T> &o,
                          explicit_make_error_code_compatible_copy_conversion_tag /*unused*/ =
                          explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>()))) // NOLINT
      : base{in_place_type<typename base::_error_type>, make_error_code(detail::extract_error_from_failure<error_type>(o))}
      , _ptr()
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<U>::value && predicate::template enable_compatible_conversion<void, T, U, void>))
  constexpr basic_outcome(const failure_type<T, U> &o, explicit_compatible_copy_conversion_tag /*unused*/ = explicit_compatible_copy_conversion_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T> &&detail::is_nothrow_constructible<exception_type, U>) // NOLINT
      : base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(o)}
      , _ptr(detail::extract_exception_from_failure<exception_type>(o))
  {
    if(!o.has_error())
    {
      this->_state._status.set_have_error(false);
    }
    if(o.has_exception())
    {
      this->_state._status.set_have_exception(true);
    }
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, T, void, void>))
  constexpr basic_outcome(failure_type<T> &&o,
                          error_failure_tag /*unused*/ = error_failure_tag()) noexcept(detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o))}
      , _ptr()
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, void, T, void>))
  constexpr basic_outcome(failure_type<T> &&o,
                          exception_failure_tag /*unused*/ = exception_failure_tag()) noexcept(detail::is_nothrow_constructible<exception_type, T>) // NOLINT
      : base()
      , _ptr(detail::extract_exception_from_failure<exception_type>(static_cast<failure_type<T> &&>(o)))
  {
    this->_state._status.set_have_exception(true);
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_make_error_code_compatible_conversion<void, T, void, void>))
  constexpr basic_outcome(failure_type<T> &&o,
                          explicit_make_error_code_compatible_move_conversion_tag /*unused*/ =
                          explicit_make_error_code_compatible_move_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>()))) // NOLINT
      : base{in_place_type<typename base::_error_type>, make_error_code(detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o)))}
      , _ptr()
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<U>::value && predicate::template enable_compatible_conversion<void, T, U, void>))
  constexpr basic_outcome(failure_type<T, U> &&o, explicit_compatible_move_conversion_tag /*unused*/ = explicit_compatible_move_conversion_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T> &&detail::is_nothrow_constructible<exception_type, U>) // NOLINT
      : base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(static_cast<failure_type<T, U> &&>(o))}
      , _ptr(detail::extract_exception_from_failure<exception_type>(static_cast<failure_type<T, U> &&>(o)))
  {
    if(!o.has_error())
    {
      this->_state._status.set_have_error(false);
    }
    if(o.has_exception())
    {
      this->_state._status.set_have_exception(true);
    }
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_move_construction(this, static_cast<failure_type<T, U> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  using base::operator==;
  using base::operator!=;
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V, class W)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<detail::devoid<value_type>>() == std::declval<detail::devoid<T>>()), //
                    OUTCOME_TEXPR(std::declval<detail::devoid<error_type>>() == std::declval<detail::devoid<U>>()), //
                    OUTCOME_TEXPR(std::declval<detail::devoid<exception_type>>() == std::declval<detail::devoid<V>>()))
  constexpr bool operator==(const basic_outcome<T, U, V, W> &o) const noexcept( //
  noexcept(std::declval<detail::devoid<value_type>>() == std::declval<detail::devoid<T>>()) //
  &&noexcept(std::declval<detail::devoid<error_type>>() == std::declval<detail::devoid<U>>()) //
  &&noexcept(std::declval<detail::devoid<exception_type>>() == std::declval<detail::devoid<V>>()))
  {
    if(this->_state._status.have_value() && o._state._status.have_value())
    {
      return this->_state._value == o._state._value; // NOLINT
    }
    if(this->_state._status.have_error() && o._state._status.have_error() //
       && this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_state._error == o._state._error && this->_ptr == o._ptr;
    }
    if(this->_state._status.have_error() && o._state._status.have_error())
    {
      return this->_state._error == o._state._error;
    }
    if(this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_ptr == o._ptr;
    }
    return false;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<error_type>() == std::declval<T>()), //
                    OUTCOME_TEXPR(std::declval<exception_type>() == std::declval<U>()))
  constexpr bool operator==(const failure_type<T, U> &o) const noexcept( //
  noexcept(std::declval<error_type>() == std::declval<T>()) &&noexcept(std::declval<exception_type>() == std::declval<U>()))
  {
    if(this->_state._status.have_error() && o._state._status.have_error() //
       && this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_state._error == o.error() && this->_ptr == o.exception();
    }
    if(this->_state._status.have_error() && o._state._status.have_error())
    {
      return this->_state._error == o.error();
    }
    if(this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_ptr == o.exception();
    }
    return false;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V, class W)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<detail::devoid<value_type>>() != std::declval<detail::devoid<T>>()), //
                    OUTCOME_TEXPR(std::declval<detail::devoid<error_type>>() != std::declval<detail::devoid<U>>()), //
                    OUTCOME_TEXPR(std::declval<detail::devoid<exception_type>>() != std::declval<detail::devoid<V>>()))
  constexpr bool operator!=(const basic_outcome<T, U, V, W> &o) const noexcept( //
  noexcept(std::declval<detail::devoid<value_type>>() != std::declval<detail::devoid<T>>()) //
  &&noexcept(std::declval<detail::devoid<error_type>>() != std::declval<detail::devoid<U>>()) //
  &&noexcept(std::declval<detail::devoid<exception_type>>() != std::declval<detail::devoid<V>>()))
  {
    if(this->_state._status.have_value() && o._state._status.have_value())
    {
      return this->_state._value != o._state._value; // NOLINT
    }
    if(this->_state._status.have_error() && o._state._status.have_error() //
       && this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_state._error != o._state._error || this->_ptr != o._ptr;
    }
    if(this->_state._status.have_error() && o._state._status.have_error())
    {
      return this->_state._error != o._state._error;
    }
    if(this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_ptr != o._ptr;
    }
    return true;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<error_type>() != std::declval<T>()), //
                    OUTCOME_TEXPR(std::declval<exception_type>() != std::declval<U>()))
  constexpr bool operator!=(const failure_type<T, U> &o) const noexcept( //
  noexcept(std::declval<error_type>() == std::declval<T>()) &&noexcept(std::declval<exception_type>() == std::declval<U>()))
  {
    if(this->_state._status.have_error() && o._state._status.have_error() //
       && this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_state._error != o.error() || this->_ptr != o.exception();
    }
    if(this->_state._status.have_error() && o._state._status.have_error())
    {
      return this->_state._error != o.error();
    }
    if(this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_ptr != o.exception();
    }
    return true;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  constexpr void swap(basic_outcome &o) noexcept((std::is_void<value_type>::value || detail::is_nothrow_swappable<value_type>::value) //
                                                 && (std::is_void<error_type>::value || detail::is_nothrow_swappable<error_type>::value) //
                                                 && (std::is_void<exception_type>::value || detail::is_nothrow_swappable<exception_type>::value))
  {
#ifdef __cpp_exceptions
    constexpr bool value_throws = !std::is_void<value_type>::value && !detail::is_nothrow_swappable<value_type>::value;
    constexpr bool error_throws = !std::is_void<error_type>::value && !detail::is_nothrow_swappable<error_type>::value;
    constexpr bool exception_throws = !std::is_void<exception_type>::value && !detail::is_nothrow_swappable<exception_type>::value;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant
#endif
    if(!exception_throws && !value_throws && !error_throws)
    {
      // Simples
      this->_state.swap(o._state);
      using std::swap;
      swap(this->_ptr, o._ptr);
      return;
    }
    struct _
    {
      basic_outcome &a, &b;
      bool exceptioned{false};
      bool all_good{false};
      ~_()
      {
        if(!all_good)
        {
          // We lost one of the values
          a._state._status.set_have_lost_consistency(true);
          b._state._status.set_have_lost_consistency(true);
          return;
        }
        if(exceptioned)
        {
          // The value + error swap threw an exception. Try to swap back _ptr
          try
          {
            strong_swap(all_good, a._ptr, b._ptr);
          }
          catch(...)
          {
            // We lost one of the values
            a._state._status.set_have_lost_consistency(true);
            b._state._status.set_have_lost_consistency(true);
            // throw away second exception
          }
          // Prevent has_value() == has_error() or has_value() == has_exception()
          auto check = [](basic_outcome *t) {
            if(t->has_value() && (t->has_error() || t->has_exception()))
            {
              t->_state._status.set_have_error(false).set_have_exception(false);
              t->_state._status.set_have_lost_consistency(true);
            }
            if(!t->has_value() && !(t->has_error() || t->has_exception()))
            {
              // Choose error, for no particular reason
              t->_state._status.set_have_error(true).set_have_lost_consistency(true);
            }
          };
          check(&a);
          check(&b);
        }
      }
    } _{*this, o};
    strong_swap(_.all_good, this->_ptr, o._ptr);
    _.exceptioned = true;
    this->_state.swap(o._state);
    _.exceptioned = false;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#else
    this->_state.swap(o._state);
    using std::swap;
    swap(this->_ptr, o._ptr);
#endif
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  failure_type<error_type, exception_type> as_failure() const &
  {
    if(this->has_error() && this->has_exception())
    {
      return failure_type<error_type, exception_type>(this->assume_error(), this->assume_exception(), hooks::spare_storage(this));
    }
    if(this->has_exception())
    {
      return failure_type<error_type, exception_type>(in_place_type<exception_type>, this->assume_exception(), hooks::spare_storage(this));
    }
    return failure_type<error_type, exception_type>(in_place_type<error_type>, this->assume_error(), hooks::spare_storage(this));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  failure_type<error_type, exception_type> as_failure() &&
  {
    this->_state._status.set_have_moved_from(true);
    if(this->has_error() && this->has_exception())
    {
      return failure_type<error_type, exception_type>(static_cast<S &&>(this->assume_error()), static_cast<P &&>(this->assume_exception()),
                                                      hooks::spare_storage(this));
    }
    if(this->has_exception())
    {
      return failure_type<error_type, exception_type>(in_place_type<exception_type>, static_cast<P &&>(this->assume_exception()), hooks::spare_storage(this));
    }
    return failure_type<error_type, exception_type>(in_place_type<error_type>, static_cast<S &&>(this->assume_error()), hooks::spare_storage(this));
  }
#ifdef __APPLE__
  failure_type<error_type, exception_type> _xcode_workaround_as_failure() &&;
#endif
};
// C++ 20 operator== rewriting should take care of this for us, indeed
// if we don't disable it, we cause Concept recursion to infinity!
#if __cplusplus < 202000L && !_HAS_CXX20
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T, class U, class V, //
                 class R, class S, class P, class N)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<basic_outcome<R, S, P, N>>() == std::declval<basic_result<T, U, V>>()))
constexpr inline bool operator==(const basic_result<T, U, V> &a, const basic_outcome<R, S, P, N> &b) noexcept( //
noexcept(std::declval<basic_outcome<R, S, P, N>>() == std::declval<basic_result<T, U, V>>()))
{
  return b == a;
}
#endif
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T, class U, class V, //
                 class R, class S, class P, class N)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<basic_outcome<R, S, P, N>>() != std::declval<basic_result<T, U, V>>()))
constexpr inline bool operator!=(const basic_result<T, U, V> &a, const basic_outcome<R, S, P, N> &b) noexcept( //
noexcept(std::declval<basic_outcome<R, S, P, N>>() != std::declval<basic_result<T, U, V>>()))
{
  return b != a;
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S, class P, class N> inline void swap(basic_outcome<R, S, P, N> &a, basic_outcome<R, S, P, N> &b) noexcept(noexcept(a.swap(b)))
{
  a.swap(b);
}
namespace hooks
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S, class P, class NoValuePolicy, class U>
  constexpr inline void override_outcome_exception(basic_outcome<R, S, P, NoValuePolicy> *o, U &&v) noexcept
  {
    o->_ptr = static_cast<U &&>(v); // NOLINT
    o->_state._status.set_have_exception(true);
  }
} // namespace hooks
OUTCOME_V2_NAMESPACE_END
#ifdef __clang__
#pragma clang diagnostic pop
#endif
/* Exception observers for outcome type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (6 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_OUTCOME_EXCEPTION_OBSERVERS_IMPL_HPP
#define OUTCOME_BASIC_OUTCOME_EXCEPTION_OBSERVERS_IMPL_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  template <class R, class S, class P, class NoValuePolicy, class Impl> inline constexpr auto &&base::_exception(Impl &&self) noexcept
  {
    // Impl will be some internal implementation class which has no knowledge of the _ptr stored
    // beneath it. So statically cast, preserving rvalue and constness, to the derived class.
    using Outcome = OUTCOME_V2_NAMESPACE::detail::rebind_type<basic_outcome<R, S, P, NoValuePolicy>, decltype(self)>;
#if defined(_MSC_VER) && _MSC_VER < 1920
    // VS2017 tries a copy construction in the correct implementation despite that Outcome is always a rvalue or lvalue ref! :(
    basic_outcome<R, S, P, NoValuePolicy> &_self = (basic_outcome<R, S, P, NoValuePolicy> &) (self); // NOLINT
#else
    Outcome _self = static_cast<Outcome>(self); // NOLINT
#endif
    return static_cast<Outcome>(_self)._ptr;
  }
} // namespace policy
namespace detail
{
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::assume_exception() & noexcept
  {
    NoValuePolicy::narrow_exception_check(*this);
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(*this);
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr const typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::assume_exception() const &noexcept
  {
    NoValuePolicy::narrow_exception_check(*this);
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(*this);
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &&basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::assume_exception() && noexcept
  {
    NoValuePolicy::narrow_exception_check(std::move(*this));
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(std::move(*this));
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr const typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &&basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::assume_exception() const &&noexcept
  {
    NoValuePolicy::narrow_exception_check(std::move(*this));
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(std::move(*this));
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception() &
  {
    NoValuePolicy::wide_exception_check(*this);
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(*this);
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr const typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception() const &
  {
    NoValuePolicy::wide_exception_check(*this);
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(*this);
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &&basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception() &&
  {
    NoValuePolicy::wide_exception_check(std::move(*this));
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(std::move(*this));
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr const typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &&basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception() const &&
  {
    NoValuePolicy::wide_exception_check(std::move(*this));
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(std::move(*this));
  }
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
#if !defined(NDEBUG)
OUTCOME_V2_NAMESPACE_BEGIN
// Check is trivial in all ways except default constructibility and standard layout
// static_assert(std::is_trivial<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivial!");
// static_assert(std::is_trivially_default_constructible<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially default
// constructible!");
static_assert(std::is_trivially_copyable<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially copyable!");
static_assert(std::is_trivially_assignable<basic_outcome<int, long, double, policy::all_narrow>, basic_outcome<int, long, double, policy::all_narrow>>::value,
              "outcome<int> is not trivially assignable!");
static_assert(std::is_trivially_destructible<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially destructible!");
static_assert(std::is_trivially_copy_constructible<basic_outcome<int, long, double, policy::all_narrow>>::value,
              "outcome<int> is not trivially copy constructible!");
static_assert(std::is_trivially_move_constructible<basic_outcome<int, long, double, policy::all_narrow>>::value,
              "outcome<int> is not trivially move constructible!");
static_assert(std::is_trivially_copy_assignable<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially copy assignable!");
static_assert(std::is_trivially_move_assignable<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially move assignable!");
// Can't be standard layout as non-static member data is defined in more than one inherited class
// static_assert(std::is_standard_layout<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not a standard layout type!");
OUTCOME_V2_NAMESPACE_END
#endif
#endif
/* Traits for Outcome
(C) 2018-2019 Niall Douglas <http://www.nedproductions.biz/> (3 commits)
File Created: March 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_TRAIT_STD_EXCEPTION_HPP
#define OUTCOME_TRAIT_STD_EXCEPTION_HPP
#include <exception>
OUTCOME_V2_NAMESPACE_BEGIN
namespace policy
{
  namespace detail
  {
    /* Pass through `make_exception_ptr` function for `std::exception_ptr`.
    */
    inline std::exception_ptr make_exception_ptr(std::exception_ptr v) { return v; }
    // Try ADL, if not use fall backs above
    template <class T> constexpr inline decltype(auto) exception_ptr(T &&v) { return make_exception_ptr(std::forward<T>(v)); }
  } // namespace detail
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T> constexpr inline decltype(auto) exception_ptr(T &&v) { return detail::exception_ptr(std::forward<T>(v)); }
  namespace detail
  {
    template <bool has_error_payload> struct _rethrow_exception
    {
      template <class Exception> explicit _rethrow_exception(Exception && /*unused*/) // NOLINT
      {
      }
    };
    template <> struct _rethrow_exception<true>
    {
      template <class Exception> explicit _rethrow_exception(Exception &&excpt) // NOLINT
      {
        // ADL
        rethrow_exception(policy::exception_ptr(std::forward<Exception>(excpt)));
      }
    };
  } // namespace detail
} // namespace policy
namespace trait
{
  namespace detail
  {
    // Shortcut this for lower build impact
    template <> struct _is_exception_ptr_available<std::exception_ptr>
    {
      static constexpr bool value = true;
      using type = std::exception_ptr;
    };
  } // namespace detail
  // std::exception_ptr is an error type
  template <> struct is_error_type<std::exception_ptr>
  {
    static constexpr bool value = true;
  };
} // namespace trait
OUTCOME_V2_NAMESPACE_END
#endif
/* A very simple result type
(C) 2018-2020 Niall Douglas <http://www.nedproductions.biz/> (11 commits)
File Created: Apr 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_EXPERIMENTAL_STATUS_RESULT_HPP
#define OUTCOME_EXPERIMENTAL_STATUS_RESULT_HPP
/* Policies for result and outcome
(C) 2018-2019 Niall Douglas <http://www.nedproductions.biz/> (4 commits)
File Created: Sep 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_FAIL_TO_COMPILE_OBSERVERS_HPP
#define OUTCOME_POLICY_FAIL_TO_COMPILE_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
#define OUTCOME_FAIL_TO_COMPILE_OBSERVERS_MESSAGE "Attempt to wide observe value, error or " "exception for a basic_result/basic_outcome given an EC or EP type which is not void, and for whom " "trait::is_error_code_available<EC>, trait::is_exception_ptr_available<EC>, and trait::is_exception_ptr_available<EP> " "are all false. Please specify a NoValuePolicy to tell basic_result/basic_outcome what to do, or else use " "a more specific convenience type alias such as unchecked<T, E> to indicate you want the wide " "observers to be narrow, or checked<T, E> to indicate you always want an exception throw etc."
namespace policy
{
  struct fail_to_compile_observers : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl && /* unused */) { static_assert(!std::is_same<Impl, Impl>::value, "Attempt to wide observe value, error or " "exception for a basic_result/basic_outcome given an EC or EP type which is not void, and for whom " "trait::is_error_code_available<EC>, trait::is_exception_ptr_available<EC>, and trait::is_exception_ptr_available<EP> " "are all false. Please specify a NoValuePolicy to tell basic_result/basic_outcome what to do, or else use " "a more specific convenience type alias such as unchecked<T, E> to indicate you want the wide " "observers to be narrow, or checked<T, E> to indicate you always want an exception throw etc."); }
    template <class Impl> static constexpr void wide_error_check(Impl && /* unused */) { static_assert(!std::is_same<Impl, Impl>::value, "Attempt to wide observe value, error or " "exception for a basic_result/basic_outcome given an EC or EP type which is not void, and for whom " "trait::is_error_code_available<EC>, trait::is_exception_ptr_available<EC>, and trait::is_exception_ptr_available<EP> " "are all false. Please specify a NoValuePolicy to tell basic_result/basic_outcome what to do, or else use " "a more specific convenience type alias such as unchecked<T, E> to indicate you want the wide " "observers to be narrow, or checked<T, E> to indicate you always want an exception throw etc."); }
    template <class Impl> static constexpr void wide_exception_check(Impl && /* unused */) { static_assert(!std::is_same<Impl, Impl>::value, "Attempt to wide observe value, error or " "exception for a basic_result/basic_outcome given an EC or EP type which is not void, and for whom " "trait::is_error_code_available<EC>, trait::is_exception_ptr_available<EC>, and trait::is_exception_ptr_available<EP> " "are all false. Please specify a NoValuePolicy to tell basic_result/basic_outcome what to do, or else use " "a more specific convenience type alias such as unchecked<T, E> to indicate you want the wide " "observers to be narrow, or checked<T, E> to indicate you always want an exception throw etc."); }
  };
} // namespace policy
#undef OUTCOME_FAIL_TO_COMPILE_OBSERVERS_MESSAGE
OUTCOME_V2_NAMESPACE_END
#endif
/* Proposed SG14 status_code
(C) 2018 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_HPP
#define SYSTEM_ERROR2_HPP
/* Proposed SG14 status_code
(C) 2018 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_ERROR_HPP
#define SYSTEM_ERROR2_ERROR_HPP
/* Proposed SG14 status_code
(C) 2018 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Jun 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_ERRORED_STATUS_CODE_HPP
#define SYSTEM_ERROR2_ERRORED_STATUS_CODE_HPP
/* Proposed SG14 status_code
(C) 2018 - 2020 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: May 2020


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_QUICK_STATUS_CODE_FROM_ENUM_HPP
#define SYSTEM_ERROR2_QUICK_STATUS_CODE_FROM_ENUM_HPP
/* Proposed SG14 status_code
(C) 2018 - 2020 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_GENERIC_CODE_HPP
#define SYSTEM_ERROR2_GENERIC_CODE_HPP
/* Proposed SG14 status_code
(C) 2018 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_STATUS_ERROR_HPP
#define SYSTEM_ERROR2_STATUS_ERROR_HPP
/* Proposed SG14 status_code
(C) 2018 - 2020 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_STATUS_CODE_HPP
#define SYSTEM_ERROR2_STATUS_CODE_HPP
/* Proposed SG14 status_code
(C) 2018 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_STATUS_CODE_DOMAIN_HPP
#define SYSTEM_ERROR2_STATUS_CODE_DOMAIN_HPP
/* Proposed SG14 status_code
(C) 2018 - 2020 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_CONFIG_HPP
#define SYSTEM_ERROR2_CONFIG_HPP
// < 0.1 each
#include <cassert>
#include <cstddef> // for size_t
#include <cstdlib> // for free
// 0.22
#include <type_traits>
// 0.29
#include <atomic>
// 0.28 (0.15 of which is exception_ptr)
#include <exception> // for std::exception
// <new> includes <exception>, <exception> includes <new>
#include <new>
// 0.01
#include <initializer_list>
#ifndef SYSTEM_ERROR2_CONSTEXPR14
#if 0L || __cplusplus >= 201400 || _MSC_VER >= 1910 /* VS2017 */
//! Defined to be `constexpr` when on C++ 14 or better compilers. Usually automatic, can be overriden.
#define SYSTEM_ERROR2_CONSTEXPR14 constexpr
#else
#define SYSTEM_ERROR2_CONSTEXPR14
#endif
#endif
#ifndef SYSTEM_ERROR2_NORETURN
#if 0L || (_HAS_CXX17 && _MSC_VER >= 1911 /* VS2017.3 */)
#define SYSTEM_ERROR2_NORETURN [[noreturn]]
#endif
#endif
#if !defined(SYSTEM_ERROR2_NORETURN)
#ifdef __has_cpp_attribute
#if __has_cpp_attribute(noreturn)
#define SYSTEM_ERROR2_NORETURN [[noreturn]]
#endif
#endif
#endif
#if !defined(SYSTEM_ERROR2_NORETURN)
#if defined(_MSC_VER)
#define SYSTEM_ERROR2_NORETURN __declspec(noreturn)
#elif defined(__GNUC__)
#define SYSTEM_ERROR2_NORETURN __attribute__((__noreturn__))
#else
#define SYSTEM_ERROR2_NORETURN
#endif
#endif
// GCCs before 7 don't grok [[noreturn]] virtual functions, and warn annoyingly
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 7
#undef SYSTEM_ERROR2_NORETURN
#define SYSTEM_ERROR2_NORETURN
#endif
#ifndef SYSTEM_ERROR2_NODISCARD
#if 0L || (_HAS_CXX17 && _MSC_VER >= 1911 /* VS2017.3 */)
#define SYSTEM_ERROR2_NODISCARD [[nodiscard]]
#endif
#endif
#ifndef SYSTEM_ERROR2_NODISCARD
#ifdef __has_cpp_attribute
#if __has_cpp_attribute(nodiscard)
#define SYSTEM_ERROR2_NODISCARD [[nodiscard]]
#endif
#elif defined(__clang__)
#define SYSTEM_ERROR2_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER)
// _Must_inspect_result_ expands into this
#define SYSTEM_ERROR2_NODISCARD __declspec("SAL_name" "(" "\"_Must_inspect_result_\"" "," "\"\"" "," "\"2\"" ")") __declspec("SAL_begin") __declspec("SAL_post") __declspec("SAL_mustInspect") __declspec("SAL_post") __declspec("SAL_checkReturn") __declspec("SAL_end")
#endif
#endif
#ifndef SYSTEM_ERROR2_NODISCARD
#define SYSTEM_ERROR2_NODISCARD
#endif
#ifndef SYSTEM_ERROR2_TRIVIAL_ABI
#if 0L || (__clang_major__ >= 7 && !defined(__APPLE__))
//! Defined to be `[[clang::trivial_abi]]` when on a new enough clang compiler. Usually automatic, can be overriden.
#define SYSTEM_ERROR2_TRIVIAL_ABI [[clang::trivial_abi]]
#else
#define SYSTEM_ERROR2_TRIVIAL_ABI
#endif
#endif
#ifndef SYSTEM_ERROR2_NAMESPACE
//! The system_error2 namespace name.
#define SYSTEM_ERROR2_NAMESPACE system_error2
//! Begins the system_error2 namespace.
#define SYSTEM_ERROR2_NAMESPACE_BEGIN namespace system_error2 {
//! Ends the system_error2 namespace.
#define SYSTEM_ERROR2_NAMESPACE_END }
#endif
//! Namespace for the library
SYSTEM_ERROR2_NAMESPACE_BEGIN
//! Namespace for user specialised traits
namespace traits
{
  /*! Specialise to true if you guarantee that a type is move bitcopying (i.e.
  its move constructor equals copying bits from old to new, old is left in a
  default constructed state, and calling the destructor on a default constructed
  instance is trivial). All trivially copyable types are move bitcopying by
  definition, and that is the unspecialised implementation.
  */
  template <class T> struct is_move_bitcopying
  {
    static constexpr bool value = std::is_trivially_copyable<T>::value;
  };
} // namespace traits
namespace detail
{
#if __cplusplus >= 201400 || _MSC_VER >= 1910 /* VS2017 */
  inline constexpr size_t cstrlen(const char *str)
  {
    const char *end = nullptr;
    for(end = str; *end != 0; ++end) // NOLINT
      ;
    return end - str;
  }
#else
  inline constexpr size_t cstrlen_(const char *str, size_t acc)
  {
    return (str[0] == 0) ? acc : cstrlen_(str + 1, acc + 1);
  }
  inline constexpr size_t cstrlen(const char *str)
  {
    return cstrlen_(str, 0);
  }
#endif
  /* A partially compliant implementation of C++20's std::bit_cast function contributed
  by Jesse Towner. TODO FIXME Replace with C++ 20 bit_cast when available.

  Our bit_cast is only guaranteed to be constexpr when both the input and output
  arguments are either integrals or enums. However, this covers most use cases
  since the vast majority of status_codes have an underlying type that is either
  an integral or enum. We still attempt a constexpr union-based type pun for non-array
  input types, which some compilers accept. For array inputs, we fall back to
  non-constexpr memmove.
  */
  template <class T> using is_integral_or_enum = std::integral_constant<bool, std::is_integral<T>::value || std::is_enum<T>::value>;
  template <class To, class From> using is_static_castable = std::integral_constant<bool, is_integral_or_enum<To>::value && is_integral_or_enum<From>::value>;
  template <class To, class From> using is_union_castable = std::integral_constant<bool, !is_static_castable<To, From>::value && !std::is_array<To>::value && !std::is_array<From>::value>;
  template <class To, class From> using is_bit_castable = std::integral_constant<bool, sizeof(To) == sizeof(From) && traits::is_move_bitcopying<To>::value && traits::is_move_bitcopying<From>::value>;
  template <class To, class From> union bit_cast_union {
    From source;
    To target;
  };
  template <class To, class From,
            typename std::enable_if< //
            is_bit_castable<To, From>::value //
            && is_static_castable<To, From>::value //
            && !is_union_castable<To, From>::value, //
            bool>::type = true> //
  constexpr To bit_cast(const From &from) noexcept
  {
    return static_cast<To>(from);
  }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
  template <class To, class From,
            typename std::enable_if< //
            is_bit_castable<To, From>::value //
            && !is_static_castable<To, From>::value //
            && is_union_castable<To, From>::value, //
            bool>::type = true> //
  constexpr To bit_cast(const From &from) noexcept
  {
    return bit_cast_union<To, From>{from}.target;
  }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
  template <class To, class From,
            typename std::enable_if< //
            is_bit_castable<To, From>::value //
            && !is_static_castable<To, From>::value //
            && !is_union_castable<To, From>::value, //
            bool>::type = true> //
  To bit_cast(const From &from) noexcept
  {
    bit_cast_union<To, From> ret;
    memmove(&ret.source, &from, sizeof(ret.source));
    return ret.target;
  }
  /* erasure_cast performs a bit_cast with additional rules to handle types
  of differing sizes. For integral & enum types, it may perform a narrowing
  or widing conversion with static_cast if necessary, before doing the final
  conversion with bit_cast. When casting to or from non-integral, non-enum
  types it may insert the value into another object with extra padding bytes
  to satisfy bit_cast's preconditions that both types have the same size. */
  template <class To, class From> using is_erasure_castable = std::integral_constant<bool, traits::is_move_bitcopying<To>::value && traits::is_move_bitcopying<From>::value>;
  template <class T, bool = std::is_enum<T>::value> struct identity_or_underlying_type
  {
    using type = T;
  };
  template <class T> struct identity_or_underlying_type<T, true>
  {
    using type = typename std::underlying_type<T>::type;
  };
  template <class OfSize, class OfSign>
  using erasure_integer_type = typename std::conditional<std::is_signed<typename identity_or_underlying_type<OfSign>::type>::value, typename std::make_signed<typename identity_or_underlying_type<OfSize>::type>::type, typename std::make_unsigned<typename identity_or_underlying_type<OfSize>::type>::type>::type;
  template <class ErasedType, std::size_t N> struct padded_erasure_object
  {
    static_assert(traits::is_move_bitcopying<ErasedType>::value, "ErasedType must be TriviallyCopyable or MoveBitcopying");
    static_assert(alignof(ErasedType) <= sizeof(ErasedType), "ErasedType must not be over-aligned");
    ErasedType value;
    char padding[N];
    constexpr explicit padded_erasure_object(const ErasedType &v) noexcept
        : value(v)
        , padding{}
    {
    }
  };
  template <class To, class From, typename std::enable_if<is_erasure_castable<To, From>::value && (sizeof(To) == sizeof(From)), bool>::type = true> constexpr To erasure_cast(const From &from) noexcept { return bit_cast<To>(from); }
  template <class To, class From, typename std::enable_if<is_erasure_castable<To, From>::value && is_static_castable<To, From>::value && (sizeof(To) < sizeof(From)), bool>::type = true> constexpr To erasure_cast(const From &from) noexcept { return static_cast<To>(bit_cast<erasure_integer_type<From, To>>(from)); }
  template <class To, class From, typename std::enable_if<is_erasure_castable<To, From>::value && is_static_castable<To, From>::value && (sizeof(To) > sizeof(From)), bool>::type = true> constexpr To erasure_cast(const From &from) noexcept { return bit_cast<To>(static_cast<erasure_integer_type<To, From>>(from)); }
  template <class To, class From, typename std::enable_if<is_erasure_castable<To, From>::value && !is_static_castable<To, From>::value && (sizeof(To) < sizeof(From)), bool>::type = true> constexpr To erasure_cast(const From &from) noexcept
  {
    return bit_cast<padded_erasure_object<To, sizeof(From) - sizeof(To)>>(from).value;
  }
  template <class To, class From, typename std::enable_if<is_erasure_castable<To, From>::value && !is_static_castable<To, From>::value && (sizeof(To) > sizeof(From)), bool>::type = true> constexpr To erasure_cast(const From &from) noexcept
  {
    return bit_cast<To>(padded_erasure_object<From, sizeof(To) - sizeof(From)>{from});
  }
} // namespace detail
SYSTEM_ERROR2_NAMESPACE_END
#ifndef SYSTEM_ERROR2_FATAL
#ifdef SYSTEM_ERROR2_NOT_POSIX
#error If SYSTEM_ERROR2_NOT_POSIX is defined, you must define your own SYSTEM_ERROR2_FATAL implementation!
#endif
#include <cstdlib> // for abort
#ifdef __APPLE__
#include <unistd.h> // for write
#endif
SYSTEM_ERROR2_NAMESPACE_BEGIN
namespace detail
{
  namespace avoid_stdio_include
  {
#if !defined(__APPLE__) && !defined(_MSC_VER)
    extern "C" ptrdiff_t write(int, const void *, size_t);
#elif defined(_MSC_VER)
    extern ptrdiff_t write(int, const void *, size_t);
#if (defined(__x86_64__) || defined(_M_X64)) || (defined(__aarch64__) || defined(_M_ARM64)) || (defined(__arm__) || defined(_M_ARM))
#pragma comment(linker, "/alternatename:?write@avoid_stdio_include@detail@system_error2@@YA_JHPEBX_K@Z=write")
#elif defined(__x86__) || defined(_M_IX86) || defined(__i386__)
#pragma comment(linker, "/alternatename:?write@avoid_stdio_include@detail@system_error2@@YAHHPBXI@Z=_write")
#else
#error Unknown architecture
#endif
#endif
  } // namespace avoid_stdio_include
  inline void do_fatal_exit(const char *msg)
  {
    using namespace avoid_stdio_include;
    write(2 /*stderr*/, msg, cstrlen(msg));
    write(2 /*stderr*/, "\n", 1);
    abort();
  }
} // namespace detail
SYSTEM_ERROR2_NAMESPACE_END
//! Prints msg to stderr, and calls `std::terminate()`. Can be overriden via predefinition.
#define SYSTEM_ERROR2_FATAL(msg) ::SYSTEM_ERROR2_NAMESPACE::detail::do_fatal_exit(msg)
#endif
#endif
#include <cstring> // for strchr
SYSTEM_ERROR2_NAMESPACE_BEGIN
/*! The main workhorse of the system_error2 library, can be typed (`status_code<DomainType>`), erased-immutable (`status_code<void>`) or erased-mutable (`status_code<erased<T>>`).

Be careful of placing these into containers! Equality and inequality operators are
*semantic* not exact. Therefore two distinct items will test true! To help prevent
surprise on this, `operator<` and `std::hash<>` are NOT implemented in order to
trap potential incorrectness. Define your own custom comparison functions for your
container which perform exact comparisons.
*/
template <class DomainType> class status_code;
class _generic_code_domain;
//! The generic code is a status code with the generic code domain, which is that of `errc` (POSIX).
using generic_code = status_code<_generic_code_domain>;
namespace detail
{
  template <class StatusCode> class indirecting_domain;
  template <class T> struct status_code_sizer
  {
    void *a;
    T b;
  };
  template <class To, class From> struct type_erasure_is_safe
  {
    static constexpr bool value = traits::is_move_bitcopying<From>::value //
                                  && (sizeof(status_code_sizer<From>) <= sizeof(status_code_sizer<To>));
  };
  /* We are severely limited by needing to retain C++ 11 compatibility when doing
  constexpr string parsing. MSVC lets you throw exceptions within a constexpr
  evaluation context when exceptions are globally disabled, but won't let you
  divide by zero, even if never evaluated, ever in constexpr. GCC and clang won't
  let you throw exceptions, ever, if exceptions are globally disabled. So let's
  use the trick of divide by zero in constexpr on GCC and clang if and only if
  exceptions are globally disabled.
  */
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdiv-by-zero"
#endif
#if defined(__cpp_exceptions) || (defined(_MSC_VER) && !defined(__clang__))
#define SYSTEM_ERROR2_FAIL_CONSTEXPR(msg) throw msg
#else
#define SYSTEM_ERROR2_FAIL_CONSTEXPR(msg) ((void) msg, 1 / 0)
#endif
  constexpr inline unsigned long long parse_hex_byte(char c) { return ('0' <= c && c <= '9') ? (c - '0') : ('a' <= c && c <= 'f') ? (10 + c - 'a') : ('A' <= c && c <= 'F') ? (10 + c - 'A') : SYSTEM_ERROR2_FAIL_CONSTEXPR("Invalid character in UUID"); }
  constexpr inline unsigned long long parse_uuid2(const char *s)
  {
    return ((parse_hex_byte(s[0]) << 0) | (parse_hex_byte(s[1]) << 4) | (parse_hex_byte(s[2]) << 8) | (parse_hex_byte(s[3]) << 12) | (parse_hex_byte(s[4]) << 16) | (parse_hex_byte(s[5]) << 20) | (parse_hex_byte(s[6]) << 24) | (parse_hex_byte(s[7]) << 28) | (parse_hex_byte(s[9]) << 32) | (parse_hex_byte(s[10]) << 36) |
            (parse_hex_byte(s[11]) << 40) | (parse_hex_byte(s[12]) << 44) | (parse_hex_byte(s[14]) << 48) | (parse_hex_byte(s[15]) << 52) | (parse_hex_byte(s[16]) << 56) | (parse_hex_byte(s[17]) << 60)) //
           ^ //
           ((parse_hex_byte(s[19]) << 0) | (parse_hex_byte(s[20]) << 4) | (parse_hex_byte(s[21]) << 8) | (parse_hex_byte(s[22]) << 12) | (parse_hex_byte(s[24]) << 16) | (parse_hex_byte(s[25]) << 20) | (parse_hex_byte(s[26]) << 24) | (parse_hex_byte(s[27]) << 28) | (parse_hex_byte(s[28]) << 32) |
            (parse_hex_byte(s[29]) << 36) | (parse_hex_byte(s[30]) << 40) | (parse_hex_byte(s[31]) << 44) | (parse_hex_byte(s[32]) << 48) | (parse_hex_byte(s[33]) << 52) | (parse_hex_byte(s[34]) << 56) | (parse_hex_byte(s[35]) << 60));
  }
  template <size_t N> constexpr inline unsigned long long parse_uuid_from_array(const char (&uuid)[N]) { return (N == 37) ? parse_uuid2(uuid) : ((N == 39) ? parse_uuid2(uuid + 1) : SYSTEM_ERROR2_FAIL_CONSTEXPR("UUID does not have correct length")); }
  template <size_t N> constexpr inline unsigned long long parse_uuid_from_pointer(const char *uuid) { return (N == 36) ? parse_uuid2(uuid) : ((N == 38) ? parse_uuid2(uuid + 1) : SYSTEM_ERROR2_FAIL_CONSTEXPR("UUID does not have correct length")); }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
  static constexpr unsigned long long test_uuid_parse = parse_uuid_from_array("430f1201-94fc-06c7-430f-120194fc06c7");
  //static constexpr unsigned long long test_uuid_parse2 = parse_uuid_from_array("x30f1201-94fc-06c7-430f-120194fc06c7");
} // namespace detail
/*! Abstract base class for a coding domain of a status code.
 */
class status_code_domain
{
  template <class DomainType> friend class status_code;
  template <class StatusCode> friend class indirecting_domain;
public:
  //! Type of the unique id for this domain.
  using unique_id_type = unsigned long long;
  /*! (Potentially thread safe) Reference to a message string.

  Be aware that you cannot add payload to implementations of this class.
  You get exactly the `void *[3]` array to keep state, this is usually
  sufficient for a `std::shared_ptr<>` or a `std::string`.

  You can install a handler to be called when this object is copied,
  moved and destructed. This takes the form of a C function pointer.
  */
  class string_ref
  {
  public:
    //! The value type
    using value_type = const char;
    //! The size type
    using size_type = size_t;
    //! The pointer type
    using pointer = const char *;
    //! The const pointer type
    using const_pointer = const char *;
    //! The iterator type
    using iterator = const char *;
    //! The const iterator type
    using const_iterator = const char *;
  protected:
    //! The operation occurring
    enum class _thunk_op
    {
      copy,
      move,
      destruct
    };
    //! The prototype of the handler function. Copies can throw, moves and destructs cannot.
    using _thunk_spec = void (*)(string_ref *dest, const string_ref *src, _thunk_op op);
#ifndef NDEBUG
  private:
    static void _checking_string_thunk(string_ref *dest, const string_ref *src, _thunk_op /*unused*/) noexcept
    {
      (void) dest;
      (void) src;
      assert(dest->_thunk == _checking_string_thunk); // NOLINT
      assert(src == nullptr || src->_thunk == _checking_string_thunk); // NOLINT
      // do nothing
    }
  protected:
#endif
    //! Pointers to beginning and end of character range
    pointer _begin{}, _end{};
    //! Three `void*` of state
    void *_state[3]{}; // at least the size of a shared_ptr
    //! Handler for when operations occur
    const _thunk_spec _thunk{nullptr};
    constexpr explicit string_ref(_thunk_spec thunk) noexcept
        : _thunk(thunk)
    {
    }
  public:
    //! Construct from a C string literal
    SYSTEM_ERROR2_CONSTEXPR14 explicit string_ref(const char *str, size_type len = static_cast<size_type>(-1), void *state0 = nullptr, void *state1 = nullptr, void *state2 = nullptr,
#ifndef NDEBUG
                                                  _thunk_spec thunk = _checking_string_thunk
#else
                                                  _thunk_spec thunk = nullptr
#endif
                                                  ) noexcept
        : _begin(str)
        , _end((len == static_cast<size_type>(-1)) ? (str + detail::cstrlen(str)) : (str + len))
        , // NOLINT
        _state{state0, state1, state2}
        , _thunk(thunk)
    {
    }
    //! Copy construct the derived implementation.
    string_ref(const string_ref &o)
        : _begin(o._begin)
        , _end(o._end)
        , _state{o._state[0], o._state[1], o._state[2]}
        , _thunk(o._thunk)
    {
      if(_thunk != nullptr)
      {
        _thunk(this, &o, _thunk_op::copy);
      }
    }
    //! Move construct the derived implementation.
    string_ref(string_ref &&o) noexcept
        : _begin(o._begin)
        , _end(o._end)
        , _state{o._state[0], o._state[1], o._state[2]}
        , _thunk(o._thunk)
    {
      if(_thunk != nullptr)
      {
        _thunk(this, &o, _thunk_op::move);
      }
    }
    //! Copy assignment
    string_ref &operator=(const string_ref &o)
    {
      if(this != &o)
      {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
        string_ref temp(static_cast<string_ref &&>(*this));
        this->~string_ref();
        try
        {
          new(this) string_ref(o); // may throw
        }
        catch(...)
        {
          new(this) string_ref(static_cast<string_ref &&>(temp));
          throw;
        }
#else
        this->~string_ref();
        new(this) string_ref(o);
#endif
      }
      return *this;
    }
    //! Move assignment
    string_ref &operator=(string_ref &&o) noexcept
    {
      if(this != &o)
      {
        this->~string_ref();
        new(this) string_ref(static_cast<string_ref &&>(o));
      }
      return *this;
    }
    //! Destruction
    ~string_ref()
    {
      if(_thunk != nullptr)
      {
        _thunk(this, nullptr, _thunk_op::destruct);
      }
      _begin = _end = nullptr;
    }
    //! Returns whether the reference is empty or not
    SYSTEM_ERROR2_NODISCARD bool empty() const noexcept { return _begin == _end; }
    //! Returns the size of the string
    size_type size() const noexcept { return _end - _begin; }
    //! Returns a null terminated C string
    const_pointer c_str() const noexcept { return _begin; }
    //! Returns a null terminated C string
    const_pointer data() const noexcept { return _begin; }
    //! Returns the beginning of the string
    iterator begin() noexcept { return _begin; }
    //! Returns the beginning of the string
    const_iterator begin() const noexcept { return _begin; }
    //! Returns the beginning of the string
    const_iterator cbegin() const noexcept { return _begin; }
    //! Returns the end of the string
    iterator end() noexcept { return _end; }
    //! Returns the end of the string
    const_iterator end() const noexcept { return _end; }
    //! Returns the end of the string
    const_iterator cend() const noexcept { return _end; }
  };
  /*! A reference counted, threadsafe reference to a message string.
   */
  class atomic_refcounted_string_ref : public string_ref
  {
    struct _allocated_msg
    {
      mutable std::atomic<unsigned> count{1};
    };
    _allocated_msg *&_msg() noexcept { return reinterpret_cast<_allocated_msg *&>(this->_state[0]); } // NOLINT
    const _allocated_msg *_msg() const noexcept { return reinterpret_cast<const _allocated_msg *>(this->_state[0]); } // NOLINT
    static void _refcounted_string_thunk(string_ref *_dest, const string_ref *_src, _thunk_op op) noexcept
    {
      auto dest = static_cast<atomic_refcounted_string_ref *>(_dest); // NOLINT
      auto src = static_cast<const atomic_refcounted_string_ref *>(_src); // NOLINT
      (void) src;
      assert(dest->_thunk == _refcounted_string_thunk); // NOLINT
      assert(src == nullptr || src->_thunk == _refcounted_string_thunk); // NOLINT
      switch(op)
      {
      case _thunk_op::copy:
      {
        if(dest->_msg() != nullptr)
        {
          auto count = dest->_msg()->count.fetch_add(1, std::memory_order_relaxed);
          (void) count;
          assert(count != 0); // NOLINT
        }
        return;
      }
      case _thunk_op::move:
      {
        assert(src); // NOLINT
        auto msrc = const_cast<atomic_refcounted_string_ref *>(src); // NOLINT
        msrc->_begin = msrc->_end = nullptr;
        msrc->_state[0] = msrc->_state[1] = msrc->_state[2] = nullptr;
        return;
      }
      case _thunk_op::destruct:
      {
        if(dest->_msg() != nullptr)
        {
          auto count = dest->_msg()->count.fetch_sub(1, std::memory_order_release);
          if(count == 1)
          {
            std::atomic_thread_fence(std::memory_order_acquire);
            free((void *) dest->_begin); // NOLINT
            delete dest->_msg(); // NOLINT
          }
        }
      }
      }
    }
  public:
    //! Construct from a C string literal allocated using `malloc()`.
    explicit atomic_refcounted_string_ref(const char *str, size_type len = static_cast<size_type>(-1), void *state1 = nullptr, void *state2 = nullptr) noexcept
        : string_ref(str, len, new(std::nothrow) _allocated_msg, state1, state2, _refcounted_string_thunk)
    {
      if(_msg() == nullptr)
      {
        free((void *) this->_begin); // NOLINT
        _msg() = nullptr; // disabled
        this->_begin = "failed to get message from system";
        this->_end = strchr(this->_begin, 0);
        return;
      }
    }
  };
private:
  unique_id_type _id;
protected:
  /*! Use [https://www.random.org/cgi-bin/randbyte?nbytes=8&format=h](https://www.random.org/cgi-bin/randbyte?nbytes=8&format=h) to get a random 64 bit id.

  Do NOT make up your own value. Do NOT use zero.
  */
  constexpr explicit status_code_domain(unique_id_type id) noexcept
      : _id(id)
  {
  }
  /*! UUID constructor, where input is constexpr parsed into a `unique_id_type`.
   */
  template <size_t N>
  constexpr explicit status_code_domain(const char (&uuid)[N]) noexcept
      : _id(detail::parse_uuid_from_array<N>(uuid))
  {
  }
  template <size_t N> struct _uuid_size
  {
  };
  //! Alternative UUID constructor
  template <size_t N>
  constexpr explicit status_code_domain(const char *uuid, _uuid_size<N> /*unused*/) noexcept
      : _id(detail::parse_uuid_from_pointer<N>(uuid))
  {
  }
  //! No public copying at type erased level
  status_code_domain(const status_code_domain &) = default;
  //! No public moving at type erased level
  status_code_domain(status_code_domain &&) = default;
  //! No public assignment at type erased level
  status_code_domain &operator=(const status_code_domain &) = default;
  //! No public assignment at type erased level
  status_code_domain &operator=(status_code_domain &&) = default;
  //! No public destruction at type erased level
  ~status_code_domain() = default;
public:
  //! True if the unique ids match.
  constexpr bool operator==(const status_code_domain &o) const noexcept { return _id == o._id; }
  //! True if the unique ids do not match.
  constexpr bool operator!=(const status_code_domain &o) const noexcept { return _id != o._id; }
  //! True if this unique is lower than the other's unique id.
  constexpr bool operator<(const status_code_domain &o) const noexcept { return _id < o._id; }
  //! Returns the unique id used to identify identical category instances.
  constexpr unique_id_type id() const noexcept { return _id; }
  //! Name of this category.
  virtual string_ref name() const noexcept = 0;
protected:
  //! True if code means failure.
  virtual bool _do_failure(const status_code<void> &code) const noexcept = 0;
  //! True if code is (potentially non-transitively) equivalent to another code in another domain.
  virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept = 0;
  //! Returns the generic code closest to this code, if any.
  virtual generic_code _generic_code(const status_code<void> &code) const noexcept = 0;
  //! Return a reference to a string textually representing a code.
  virtual string_ref _do_message(const status_code<void> &code) const noexcept = 0;
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || 0L
  //! Throw a code as a C++ exception.
  SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const = 0;
#else
  // Keep a vtable slot for binary compatibility
  SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> & /*code*/) const { abort(); }
#endif
  // For a `status_code<erased<T>>` only, copy from `src` to `dst`. Default implementation uses `memcpy()`.
  virtual void _do_erased_copy(status_code<void> &dst, const status_code<void> &src, size_t bytes) const { memcpy(&dst, &src, bytes); } // NOLINT
  // For a `status_code<erased<T>>` only, destroy the erased value type. Default implementation does nothing.
  virtual void _do_erased_destroy(status_code<void> &code, size_t bytes) const noexcept // NOLINT
  {
    (void) code;
    (void) bytes;
  }
};
SYSTEM_ERROR2_NAMESPACE_END
#endif
#if (__cplusplus >= 201700 || _HAS_CXX17) && !defined(SYSTEM_ERROR2_DISABLE_STD_IN_PLACE)
// 0.26
#include <utility> // for in_place
SYSTEM_ERROR2_NAMESPACE_BEGIN
using in_place_t = std::in_place_t;
using std::in_place;
SYSTEM_ERROR2_NAMESPACE_END
#else
SYSTEM_ERROR2_NAMESPACE_BEGIN
//! Aliases `std::in_place_t` if on C++ 17 or later, else defined locally.
struct in_place_t
{
  explicit in_place_t() = default;
};
//! Aliases `std::in_place` if on C++ 17 or later, else defined locally.
constexpr in_place_t in_place{};
SYSTEM_ERROR2_NAMESPACE_END
#endif
SYSTEM_ERROR2_NAMESPACE_BEGIN
//! Namespace for user injected mixins
namespace mixins
{
  template <class Base, class T> struct mixin : public Base
  {
    using Base::Base;
  };
} // namespace mixins
/*! A tag for an erased value type for `status_code<D>`.
Available only if `ErasedType` satisfies `traits::is_move_bitcopying<ErasedType>::value`.
*/
template <class ErasedType, //
          typename std::enable_if<traits::is_move_bitcopying<ErasedType>::value, bool>::type = true>
struct erased
{
  using value_type = ErasedType;
};
/*! Specialise this template to quickly wrap a third party enumeration into a
custom status code domain.

Use like this:

```c++
SYSTEM_ERROR2_NAMESPACE_BEGIN
template <> struct quick_status_code_from_enum<AnotherCode> : quick_status_code_from_enum_defaults<AnotherCode>
{
  // Text name of the enum
  static constexpr const auto domain_name = "Another Code";
  // Unique UUID for the enum. PLEASE use https://www.random.org/cgi-bin/randbyte?nbytes=16&format=h
  static constexpr const auto domain_uuid = "{be201f65-3962-dd0e-1266-a72e63776a42}";
  // Map of each enum value to its text string, and list of semantically equivalent errc's
  static const std::initializer_list<mapping> &value_mappings()
  {
    static const std::initializer_list<mapping<AnotherCode>> v = {
    // Format is: { enum value, "string representation", { list of errc mappings ... } }
    {AnotherCode::success1, "Success 1", {errc::success}},        //
    {AnotherCode::goaway, "Go away", {errc::permission_denied}},  //
    {AnotherCode::success2, "Success 2", {errc::success}},        //
    {AnotherCode::error2, "Error 2", {}},                         //
    };
    return v;
  }
  // Completely optional definition of mixin for the status code synthesised from `Enum`. It can be omitted.
  template <class Base> struct mixin : Base
  {
    using Base::Base;
    constexpr int custom_method() const { return 42; }
  };
};
SYSTEM_ERROR2_NAMESPACE_END
```

Note that if the `errc` mapping contains `errc::success`, then
the enumeration value is considered to be a successful value.
Otherwise it is considered to be a failure value.

The first value in the `errc` mapping is the one chosen as the
`generic_code` conversion. Other values are used during equivalence
comparisons.
*/
template <class Enum> struct quick_status_code_from_enum;
namespace detail
{
  template <class T> struct is_status_code
  {
    static constexpr bool value = false;
  };
  template <class T> struct is_status_code<status_code<T>>
  {
    static constexpr bool value = true;
  };
  template <class T> struct is_erased_status_code
  {
    static constexpr bool value = false;
  };
  template <class T> struct is_erased_status_code<status_code<erased<T>>>
  {
    static constexpr bool value = true;
  };
 #if !defined(__GNUC__) || defined(__clang__) || __GNUC__ >= 8
  // From http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4436.pdf
  namespace impl
  {
    template <typename... Ts> struct make_void
    {
      using type = void;
    };
    template <typename... Ts> using void_t = typename make_void<Ts...>::type;
    template <class...> struct types
    {
      using type = types;
    };
    template <template <class...> class T, class types, class = void> struct test_apply
    {
      using type = void;
    };
    template <template <class...> class T, class... Ts> struct test_apply<T, types<Ts...>, void_t<T<Ts...>>>
    {
      using type = T<Ts...>;
    };
  }  // namespace impl
  template <template <class...> class T, class... Ts> using test_apply = impl::test_apply<T, impl::types<Ts...>>;

  template <class T, class... Args> using get_make_status_code_result = decltype(make_status_code(std::declval<T>(), std::declval<Args>()...));
  template <class... Args> using safe_get_make_status_code_result = test_apply<get_make_status_code_result, Args...>;

#else

  // ICE avoidance form for GCCs before 8. Note this form doesn't prevent recursive make_status_code ADL instantation,
  // so in certain corner cases this will break. On the other hand, more useful than an ICE.
  namespace impl
  {
    template <typename... Ts> struct make_void
    {
      using type = void;
    };
    template <typename... Ts> using void_t = typename make_void<Ts...>::type;
    template <class...> struct types
    {
      using type = types;
    };
    template <typename Types, typename = void> struct make_status_code_rettype
    {
      using type = void;
    };
    template <typename... Args> using get_make_status_code_result = decltype(make_status_code(std::declval<Args>()...));
    template <typename... Args> struct make_status_code_rettype<types<Args...>, void_t<get_make_status_code_result<Args...>>>
    {
      using type = get_make_status_code_result<Args...>;
    };
  }  // namespace impl
  template <class... Args> struct safe_get_make_status_code_result
  {
    using type = typename impl::make_status_code_rettype<impl::types<Args...>>::type;
  };
#endif
} // namespace detail
//! Trait returning true if the type is a status code.
template <class T> struct is_status_code
{
  static constexpr bool value = detail::is_status_code<typename std::decay<T>::type>::value || detail::is_erased_status_code<typename std::decay<T>::type>::value;
};
/*! A type erased lightweight status code reflecting empty, success, or failure.
Differs from `status_code<erased<>>` by being always available irrespective of
the domain's value type, but cannot be copied, moved, nor destructed. Thus one
always passes this around by const lvalue reference.
*/
template <> class SYSTEM_ERROR2_TRIVIAL_ABI status_code<void>
{
  template <class T> friend class status_code;
public:
  //! The type of the domain.
  using domain_type = void;
  //! The type of the status code.
  using value_type = void;
  //! The type of a reference to a message string.
  using string_ref = typename status_code_domain::string_ref;
protected:
  const status_code_domain *_domain{nullptr};
protected:
  //! No default construction at type erased level
  status_code() = default;
  //! No public copying at type erased level
  status_code(const status_code &) = default;
  //! No public moving at type erased level
  status_code(status_code &&) = default;
  //! No public assignment at type erased level
  status_code &operator=(const status_code &) = default;
  //! No public assignment at type erased level
  status_code &operator=(status_code &&) = default;
  //! No public destruction at type erased level
  ~status_code() = default;
  //! Used to construct a non-empty type erased status code
  constexpr explicit status_code(const status_code_domain *v) noexcept
      : _domain(v)
  {
  }
  // Used to work around triggering a ubsan failure. Do NOT remove!
  constexpr const status_code_domain *_domain_ptr() const noexcept { return _domain; }
public:
  //! Return the status code domain.
  constexpr const status_code_domain &domain() const noexcept { return *_domain; }
  //! True if the status code is empty.
  SYSTEM_ERROR2_NODISCARD constexpr bool empty() const noexcept { return _domain == nullptr; }
  //! Return a reference to a string textually representing a code.
  string_ref message() const noexcept
  {
    // Avoid MSVC's buggy ternary operator for expensive to destruct things
    if(_domain != nullptr)
    {
      return _domain->_do_message(*this);
    }
    return string_ref("(empty)");
  }
  //! True if code means success.
  bool success() const noexcept { return (_domain != nullptr) ? !_domain->_do_failure(*this) : false; }
  //! True if code means failure.
  bool failure() const noexcept { return (_domain != nullptr) ? _domain->_do_failure(*this) : false; }
  /*! True if code is strictly (and potentially non-transitively) semantically equivalent to another code in another domain.
  Note that usually non-semantic i.e. pure value comparison is used when the other status code has the same domain.
  As `equivalent()` will try mapping to generic code, this usually captures when two codes have the same semantic
  meaning in `equivalent()`.
  */
  template <class T> bool strictly_equivalent(const status_code<T> &o) const noexcept
  {
    if(_domain && o._domain)
    {
      return _domain->_do_equivalent(*this, o);
    }
    // If we are both empty, we are equivalent
    if(!_domain && !o._domain)
    {
      return true; // NOLINT
    }
    // Otherwise not equivalent
    return false;
  }
  /*! True if code is equivalent, by any means, to another code in another domain (guaranteed transitive).
  Firstly `strictly_equivalent()` is run in both directions. If neither succeeds, each domain is asked
  for the equivalent generic code and those are compared.
  */
  template <class T> inline bool equivalent(const status_code<T> &o) const noexcept;
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || 0L
  //! Throw a code as a C++ exception.
  SYSTEM_ERROR2_NORETURN void throw_exception() const
  {
    _domain->_do_throw_exception(*this);
    abort(); // suppress buggy GCC warning
  }
#endif
};
namespace detail
{
  template <class DomainType> struct get_domain_value_type
  {
    using domain_type = DomainType;
    using value_type = typename domain_type::value_type;
  };
  template <class ErasedType> struct get_domain_value_type<erased<ErasedType>>
  {
    using domain_type = status_code_domain;
    using value_type = ErasedType;
  };
  template <class DomainType> class SYSTEM_ERROR2_TRIVIAL_ABI status_code_storage : public status_code<void>
  {
    using _base = status_code<void>;
  public:
    //! The type of the domain.
    using domain_type = typename get_domain_value_type<DomainType>::domain_type;
    //! The type of the status code.
    using value_type = typename get_domain_value_type<DomainType>::value_type;
    //! The type of a reference to a message string.
    using string_ref = typename domain_type::string_ref;
#ifndef NDEBUG
    static_assert(std::is_move_constructible<value_type>::value || std::is_copy_constructible<value_type>::value, "DomainType::value_type is neither move nor copy constructible!");
    static_assert(!std::is_default_constructible<value_type>::value || std::is_nothrow_default_constructible<value_type>::value, "DomainType::value_type is not nothrow default constructible!");
    static_assert(!std::is_move_constructible<value_type>::value || std::is_nothrow_move_constructible<value_type>::value, "DomainType::value_type is not nothrow move constructible!");
    static_assert(std::is_nothrow_destructible<value_type>::value, "DomainType::value_type is not nothrow destructible!");
#endif
    // Replace the type erased implementations with type aware implementations for better codegen
    //! Return the status code domain.
    constexpr const domain_type &domain() const noexcept { return *static_cast<const domain_type *>(this->_domain); }
    //! Reset the code to empty.
    SYSTEM_ERROR2_CONSTEXPR14 void clear() noexcept
    {
      this->_value.~value_type();
      this->_domain = nullptr;
      new(&this->_value) value_type();
    }
#if __cplusplus >= 201400 || _MSC_VER >= 1910 /* VS2017 */
    //! Return a reference to the `value_type`.
    constexpr value_type &value() &noexcept { return this->_value; }
    //! Return a reference to the `value_type`.
    constexpr value_type &&value() &&noexcept { return static_cast<value_type &&>(this->_value); }
#endif
    //! Return a reference to the `value_type`.
    constexpr const value_type &value() const &noexcept { return this->_value; }
    //! Return a reference to the `value_type`.
    constexpr const value_type &&value() const &&noexcept { return static_cast<const value_type &&>(this->_value); }
  protected:
    status_code_storage() = default;
    status_code_storage(const status_code_storage &) = default;
    SYSTEM_ERROR2_CONSTEXPR14 status_code_storage(status_code_storage &&o) noexcept
        : _base(static_cast<status_code_storage &&>(o))
        , _value(static_cast<status_code_storage &&>(o)._value)
    {
      o._domain = nullptr;
    }
    status_code_storage &operator=(const status_code_storage &) = default;
    SYSTEM_ERROR2_CONSTEXPR14 status_code_storage &operator=(status_code_storage &&o) noexcept
    {
      this->~status_code_storage();
      new(this) status_code_storage(static_cast<status_code_storage &&>(o));
      return *this;
    }
    ~status_code_storage() = default;
    value_type _value{};
    struct _value_type_constructor
    {
    };
    template <class... Args>
    constexpr status_code_storage(_value_type_constructor /*unused*/, const status_code_domain *v, Args &&... args)
        : _base(v)
        , _value(static_cast<Args &&>(args)...)
    {
    }
  };
} // namespace detail
/*! A lightweight, typed, status code reflecting empty, success, or failure.
This is the main workhorse of the system_error2 library. Its characteristics reflect the value type
set by its domain type, so if that value type is move-only or trivial, so is this.

An ADL discovered helper function `make_status_code(T, Args...)` is looked up by one of the constructors.
If it is found, and it generates a status code compatible with this status code, implicit construction
is made available.

You may mix in custom member functions and member function overrides by injecting a specialisation of
`mixins::mixin<Base, YourDomainType>`. Your mixin must inherit from `Base`.
*/
template <class DomainType> class SYSTEM_ERROR2_TRIVIAL_ABI status_code : public mixins::mixin<detail::status_code_storage<DomainType>, DomainType>
{
  template <class T> friend class status_code;
  using _base = mixins::mixin<detail::status_code_storage<DomainType>, DomainType>;
public:
  //! The type of the domain.
  using domain_type = DomainType;
  //! The type of the status code.
  using value_type = typename domain_type::value_type;
  //! The type of a reference to a message string.
  using string_ref = typename domain_type::string_ref;
protected:
  using _base::_base;
public:
  //! Default construction to empty
  status_code() = default;
  //! Copy constructor
  status_code(const status_code &) = default;
  //! Move constructor
  status_code(status_code &&) = default; // NOLINT
  //! Copy assignment
  status_code &operator=(const status_code &) = default;
  //! Move assignment
  status_code &operator=(status_code &&) = default; // NOLINT
  ~status_code() = default;
  //! Return a copy of the code.
  SYSTEM_ERROR2_CONSTEXPR14 status_code clone() const { return *this; }
  /***** KEEP THESE IN SYNC WITH ERRORED_STATUS_CODE *****/
  //! Implicit construction from any type where an ADL discovered `make_status_code(T, Args ...)` returns a `status_code`.
  template <class T, class... Args, //
            class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<T, Args...>::type, // Safe ADL lookup of make_status_code(), returns void if not found
            typename std::enable_if<!std::is_same<typename std::decay<T>::type, status_code>::value // not copy/move of self
                                    && !std::is_same<typename std::decay<T>::type, in_place_t>::value // not in_place_t
                                    && is_status_code<MakeStatusCodeResult>::value // ADL makes a status code
                                    && std::is_constructible<status_code, MakeStatusCodeResult>::value, // ADLed status code is compatible
                                    bool>::type = true>
  constexpr status_code(T &&v, Args &&... args) noexcept(noexcept(make_status_code(std::declval<T>(), std::declval<Args>()...))) // NOLINT
      : status_code(make_status_code(static_cast<T &&>(v), static_cast<Args &&>(args)...))
  {
  }
  //! Implicit construction from any `quick_status_code_from_enum<Enum>` enumerated type.
  template <class Enum, //
            class QuickStatusCodeType = typename quick_status_code_from_enum<Enum>::code_type, // Enumeration has been activated
            typename std::enable_if<std::is_constructible<status_code, QuickStatusCodeType>::value, // Its status code is compatible
                                    bool>::type = true>
  constexpr status_code(Enum &&v) noexcept(std::is_nothrow_constructible<status_code, QuickStatusCodeType>::value) // NOLINT
      : status_code(QuickStatusCodeType(static_cast<Enum &&>(v)))
  {
  }
  //! Explicit in-place construction. Disables if `domain_type::get()` is not a valid expression.
  template <class... Args>
  constexpr explicit status_code(in_place_t /*unused */, Args &&... args) noexcept(std::is_nothrow_constructible<value_type, Args &&...>::value)
      : _base(typename _base::_value_type_constructor{}, &domain_type::get(), static_cast<Args &&>(args)...)
  {
  }
  //! Explicit in-place construction from initialiser list. Disables if `domain_type::get()` is not a valid expression.
  template <class T, class... Args>
  constexpr explicit status_code(in_place_t /*unused */, std::initializer_list<T> il, Args &&... args) noexcept(std::is_nothrow_constructible<value_type, std::initializer_list<T>, Args &&...>::value)
      : _base(typename _base::_value_type_constructor{}, &domain_type::get(), il, static_cast<Args &&>(args)...)
  {
  }
  //! Explicit copy construction from a `value_type`. Disables if `domain_type::get()` is not a valid expression.
  constexpr explicit status_code(const value_type &v) noexcept(std::is_nothrow_copy_constructible<value_type>::value)
      : _base(typename _base::_value_type_constructor{}, &domain_type::get(), v)
  {
  }
  //! Explicit move construction from a `value_type`. Disables if `domain_type::get()` is not a valid expression.
  constexpr explicit status_code(value_type &&v) noexcept(std::is_nothrow_move_constructible<value_type>::value)
      : _base(typename _base::_value_type_constructor{}, &domain_type::get(), static_cast<value_type &&>(v))
  {
  }
  /*! Explicit construction from an erased status code. Available only if
  `value_type` is trivially copyable or move bitcopying, and `sizeof(status_code) <= sizeof(status_code<erased<>>)`.
  Does not check if domains are equal.
  */
  template <class ErasedType, //
            typename std::enable_if<detail::type_erasure_is_safe<ErasedType, value_type>::value, bool>::type = true>
  constexpr explicit status_code(const status_code<erased<ErasedType>> &v) noexcept(std::is_nothrow_copy_constructible<value_type>::value)
      : status_code(detail::erasure_cast<value_type>(v.value()))
  {
#if __cplusplus >= 201400
    assert(v.domain() == this->domain());
#endif
  }
  //! Return a reference to a string textually representing a code.
  string_ref message() const noexcept
  {
    // Avoid MSVC's buggy ternary operator for expensive to destruct things
    if(this->_domain != nullptr)
    {
      return string_ref(this->domain()._do_message(*this));
    }
    return string_ref("(empty)");
  }
};
namespace traits
{
  template <class DomainType> struct is_move_bitcopying<status_code<DomainType>>
  {
    static constexpr bool value = is_move_bitcopying<typename DomainType::value_type>::value;
  };
} // namespace traits
/*! Type erased, move-only status_code, unlike `status_code<void>` which cannot be moved nor destroyed. Available
only if `erased<>` is available, which is when the domain's type is trivially
copyable or is move relocatable, and if the size of the domain's typed error code is less than or equal to
this erased error code. Copy construction is disabled, but if you want a copy call `.clone()`.

An ADL discovered helper function `make_status_code(T, Args...)` is looked up by one of the constructors.
If it is found, and it generates a status code compatible with this status code, implicit construction
is made available.
*/
template <class ErasedType> class SYSTEM_ERROR2_TRIVIAL_ABI status_code<erased<ErasedType>> : public mixins::mixin<detail::status_code_storage<erased<ErasedType>>, erased<ErasedType>>
{
  template <class T> friend class status_code;
  using _base = mixins::mixin<detail::status_code_storage<erased<ErasedType>>, erased<ErasedType>>;
public:
  //! The type of the domain (void, as it is erased).
  using domain_type = void;
  //! The type of the erased status code.
  using value_type = ErasedType;
  //! The type of a reference to a message string.
  using string_ref = typename _base::string_ref;
public:
  //! Default construction to empty
  status_code() = default;
  //! Copy constructor
  status_code(const status_code &) = delete;
  //! Move constructor
  status_code(status_code &&) = default; // NOLINT
  //! Copy assignment
  status_code &operator=(const status_code &) = delete;
  //! Move assignment
  status_code &operator=(status_code &&) = default; // NOLINT
  ~status_code()
  {
    if(nullptr != this->_domain)
    {
      this->_domain->_do_erased_destroy(*this, sizeof(*this));
    }
  }
  //! Return a copy of the erased code by asking the domain to perform the erased copy.
  status_code clone() const
  {
    if(nullptr == this->_domain)
    {
      return {};
    }
    status_code x;
    this->_domain->_do_erased_copy(x, *this, sizeof(*this));
    return x;
  }
  /***** KEEP THESE IN SYNC WITH ERRORED_STATUS_CODE *****/
  //! Implicit copy construction from any other status code if its value type is trivially copyable and it would fit into our storage
  template <class DomainType, //
            typename std::enable_if<std::is_trivially_copyable<typename DomainType::value_type>::value //
                                    && detail::type_erasure_is_safe<value_type, typename DomainType::value_type>::value,
                                    bool>::type = true>
  constexpr status_code(const status_code<DomainType> &v) noexcept // NOLINT
      : _base(typename _base::_value_type_constructor{}, v._domain_ptr(), detail::erasure_cast<value_type>(v.value()))
  {
  }
  //! Implicit move construction from any other status code if its value type is trivially copyable or move bitcopying and it would fit into our storage
  template <class DomainType, //
            typename std::enable_if<detail::type_erasure_is_safe<value_type, typename DomainType::value_type>::value, bool>::type = true>
  SYSTEM_ERROR2_CONSTEXPR14 status_code(status_code<DomainType> &&v) noexcept // NOLINT
      : _base(typename _base::_value_type_constructor{}, v._domain_ptr(), detail::erasure_cast<value_type>(v.value()))
  {
    v._domain = nullptr;
  }
  //! Implicit construction from any type where an ADL discovered `make_status_code(T, Args ...)` returns a `status_code`.
  template <class T, class... Args, //
            class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<T, Args...>::type, // Safe ADL lookup of make_status_code(), returns void if not found
            typename std::enable_if<!std::is_same<typename std::decay<T>::type, status_code>::value // not copy/move of self
                                    && !std::is_same<typename std::decay<T>::type, value_type>::value // not copy/move of value type
                                    && is_status_code<MakeStatusCodeResult>::value // ADL makes a status code
                                    && std::is_constructible<status_code, MakeStatusCodeResult>::value, // ADLed status code is compatible
                                    bool>::type = true>
  constexpr status_code(T &&v, Args &&... args) noexcept(noexcept(make_status_code(std::declval<T>(), std::declval<Args>()...))) // NOLINT
      : status_code(make_status_code(static_cast<T &&>(v), static_cast<Args &&>(args)...))
  {
  }
  //! Implicit construction from any `quick_status_code_from_enum<Enum>` enumerated type.
  template <class Enum, //
            class QuickStatusCodeType = typename quick_status_code_from_enum<Enum>::code_type, // Enumeration has been activated
            typename std::enable_if<std::is_constructible<status_code, QuickStatusCodeType>::value, // Its status code is compatible
                                    bool>::type = true>
  constexpr status_code(Enum &&v) noexcept(std::is_nothrow_constructible<status_code, QuickStatusCodeType>::value) // NOLINT
      : status_code(QuickStatusCodeType(static_cast<Enum &&>(v)))
  {
  }
  /**** By rights ought to be removed in any formal standard ****/
  //! Reset the code to empty.
  SYSTEM_ERROR2_CONSTEXPR14 void clear() noexcept { *this = status_code(); }
  //! Return the erased `value_type` by value.
  constexpr value_type value() const noexcept { return this->_value; }
};
namespace traits
{
  template <class ErasedType> struct is_move_bitcopying<status_code<erased<ErasedType>>>
  {
    static constexpr bool value = true;
  };
} // namespace traits
SYSTEM_ERROR2_NAMESPACE_END
#endif
#include <exception> // for std::exception
SYSTEM_ERROR2_NAMESPACE_BEGIN
/*! Exception type representing a thrown status_code
*/
template <class DomainType> class status_error;
/*! The erased type edition of status_error.
*/
template <> class status_error<void> : public std::exception
{
protected:
  //! Constructs an instance. Not publicly available.
  status_error() = default;
  //! Copy constructor. Not publicly available
  status_error(const status_error &) = default;
  //! Move constructor. Not publicly available
  status_error(status_error &&) = default;
  //! Copy assignment. Not publicly available
  status_error &operator=(const status_error &) = default;
  //! Move assignment. Not publicly available
  status_error &operator=(status_error &&) = default;
  //! Destructor. Not publicly available.
  ~status_error() override = default;
public:
  //! The type of the status domain
  using domain_type = void;
  //! The type of the status code
  using status_code_type = status_code<void>;
};
/*! Exception type representing a thrown status_code
*/
template <class DomainType> class status_error : public status_error<void>
{
  status_code<DomainType> _code;
  typename DomainType::string_ref _msgref;
public:
  //! The type of the status domain
  using domain_type = DomainType;
  //! The type of the status code
  using status_code_type = status_code<DomainType>;
  //! Constructs an instance
  explicit status_error(status_code<DomainType> code)
      : _code(static_cast<status_code<DomainType> &&>(code))
      , _msgref(_code.message())
  {
  }
  //! Return an explanatory string
  virtual const char *what() const noexcept override { return _msgref.c_str(); } // NOLINT
  //! Returns a reference to the code
  const status_code_type &code() const & { return _code; }
  //! Returns a reference to the code
  status_code_type &code() & { return _code; }
  //! Returns a reference to the code
  const status_code_type &&code() const && { return _code; }
  //! Returns a reference to the code
  status_code_type &&code() && { return _code; }
};
SYSTEM_ERROR2_NAMESPACE_END
#endif
#include <cerrno> // for error constants
SYSTEM_ERROR2_NAMESPACE_BEGIN
//! The generic error coding (POSIX)
enum class errc : int
{
  success = 0,
  unknown = -1,
  address_family_not_supported = EAFNOSUPPORT,
  address_in_use = EADDRINUSE,
  address_not_available = EADDRNOTAVAIL,
  already_connected = EISCONN,
  argument_list_too_long = E2BIG,
  argument_out_of_domain = EDOM,
  bad_address = EFAULT,
  bad_file_descriptor = EBADF,
  bad_message = EBADMSG,
  broken_pipe = EPIPE,
  connection_aborted = ECONNABORTED,
  connection_already_in_progress = EALREADY,
  connection_refused = ECONNREFUSED,
  connection_reset = ECONNRESET,
  cross_device_link = EXDEV,
  destination_address_required = EDESTADDRREQ,
  device_or_resource_busy = EBUSY,
  directory_not_empty = ENOTEMPTY,
  executable_format_error = ENOEXEC,
  file_exists = EEXIST,
  file_too_large = EFBIG,
  filename_too_long = ENAMETOOLONG,
  function_not_supported = ENOSYS,
  host_unreachable = EHOSTUNREACH,
  identifier_removed = EIDRM,
  illegal_byte_sequence = EILSEQ,
  inappropriate_io_control_operation = ENOTTY,
  interrupted = EINTR,
  invalid_argument = EINVAL,
  invalid_seek = ESPIPE,
  io_error = EIO,
  is_a_directory = EISDIR,
  message_size = EMSGSIZE,
  network_down = ENETDOWN,
  network_reset = ENETRESET,
  network_unreachable = ENETUNREACH,
  no_buffer_space = ENOBUFS,
  no_child_process = ECHILD,
  no_link = ENOLINK,
  no_lock_available = ENOLCK,
  no_message = ENOMSG,
  no_protocol_option = ENOPROTOOPT,
  no_space_on_device = ENOSPC,
  no_stream_resources = ENOSR,
  no_such_device_or_address = ENXIO,
  no_such_device = ENODEV,
  no_such_file_or_directory = ENOENT,
  no_such_process = ESRCH,
  not_a_directory = ENOTDIR,
  not_a_socket = ENOTSOCK,
  not_a_stream = ENOSTR,
  not_connected = ENOTCONN,
  not_enough_memory = ENOMEM,
  not_supported = ENOTSUP,
  operation_canceled = ECANCELED,
  operation_in_progress = EINPROGRESS,
  operation_not_permitted = EPERM,
  operation_not_supported = EOPNOTSUPP,
  operation_would_block = EWOULDBLOCK,
  owner_dead = EOWNERDEAD,
  permission_denied = EACCES,
  protocol_error = EPROTO,
  protocol_not_supported = EPROTONOSUPPORT,
  read_only_file_system = EROFS,
  resource_deadlock_would_occur = EDEADLK,
  resource_unavailable_try_again = EAGAIN,
  result_out_of_range = ERANGE,
  state_not_recoverable = ENOTRECOVERABLE,
  stream_timeout = ETIME,
  text_file_busy = ETXTBSY,
  timed_out = ETIMEDOUT,
  too_many_files_open_in_system = ENFILE,
  too_many_files_open = EMFILE,
  too_many_links = EMLINK,
  too_many_symbolic_link_levels = ELOOP,
  value_too_large = EOVERFLOW,
  wrong_protocol_type = EPROTOTYPE
};
namespace detail
{
  SYSTEM_ERROR2_CONSTEXPR14 inline const char *generic_code_message(errc code) noexcept
  {
    switch(code)
    {
    case errc::success:
      return "Success";
    case errc::address_family_not_supported:
      return "Address family not supported by protocol";
    case errc::address_in_use:
      return "Address already in use";
    case errc::address_not_available:
      return "Cannot assign requested address";
    case errc::already_connected:
      return "Transport endpoint is already connected";
    case errc::argument_list_too_long:
      return "Argument list too long";
    case errc::argument_out_of_domain:
      return "Numerical argument out of domain";
    case errc::bad_address:
      return "Bad address";
    case errc::bad_file_descriptor:
      return "Bad file descriptor";
    case errc::bad_message:
      return "Bad message";
    case errc::broken_pipe:
      return "Broken pipe";
    case errc::connection_aborted:
      return "Software caused connection abort";
    case errc::connection_already_in_progress:
      return "Operation already in progress";
    case errc::connection_refused:
      return "Connection refused";
    case errc::connection_reset:
      return "Connection reset by peer";
    case errc::cross_device_link:
      return "Invalid cross-device link";
    case errc::destination_address_required:
      return "Destination address required";
    case errc::device_or_resource_busy:
      return "Device or resource busy";
    case errc::directory_not_empty:
      return "Directory not empty";
    case errc::executable_format_error:
      return "Exec format error";
    case errc::file_exists:
      return "File exists";
    case errc::file_too_large:
      return "File too large";
    case errc::filename_too_long:
      return "File name too long";
    case errc::function_not_supported:
      return "Function not implemented";
    case errc::host_unreachable:
      return "No route to host";
    case errc::identifier_removed:
      return "Identifier removed";
    case errc::illegal_byte_sequence:
      return "Invalid or incomplete multibyte or wide character";
    case errc::inappropriate_io_control_operation:
      return "Inappropriate ioctl for device";
    case errc::interrupted:
      return "Interrupted system call";
    case errc::invalid_argument:
      return "Invalid argument";
    case errc::invalid_seek:
      return "Illegal seek";
    case errc::io_error:
      return "Input/output error";
    case errc::is_a_directory:
      return "Is a directory";
    case errc::message_size:
      return "Message too long";
    case errc::network_down:
      return "Network is down";
    case errc::network_reset:
      return "Network dropped connection on reset";
    case errc::network_unreachable:
      return "Network is unreachable";
    case errc::no_buffer_space:
      return "No buffer space available";
    case errc::no_child_process:
      return "No child processes";
    case errc::no_link:
      return "Link has been severed";
    case errc::no_lock_available:
      return "No locks available";
    case errc::no_message:
      return "No message of desired type";
    case errc::no_protocol_option:
      return "Protocol not available";
    case errc::no_space_on_device:
      return "No space left on device";
    case errc::no_stream_resources:
      return "Out of streams resources";
    case errc::no_such_device_or_address:
      return "No such device or address";
    case errc::no_such_device:
      return "No such device";
    case errc::no_such_file_or_directory:
      return "No such file or directory";
    case errc::no_such_process:
      return "No such process";
    case errc::not_a_directory:
      return "Not a directory";
    case errc::not_a_socket:
      return "Socket operation on non-socket";
    case errc::not_a_stream:
      return "Device not a stream";
    case errc::not_connected:
      return "Transport endpoint is not connected";
    case errc::not_enough_memory:
      return "Cannot allocate memory";
#if ENOTSUP != EOPNOTSUPP
    case errc::not_supported:
      return "Operation not supported";
#endif
    case errc::operation_canceled:
      return "Operation canceled";
    case errc::operation_in_progress:
      return "Operation now in progress";
    case errc::operation_not_permitted:
      return "Operation not permitted";
    case errc::operation_not_supported:
      return "Operation not supported";
#if EAGAIN != EWOULDBLOCK
    case errc::operation_would_block:
      return "Resource temporarily unavailable";
#endif
    case errc::owner_dead:
      return "Owner died";
    case errc::permission_denied:
      return "Permission denied";
    case errc::protocol_error:
      return "Protocol error";
    case errc::protocol_not_supported:
      return "Protocol not supported";
    case errc::read_only_file_system:
      return "Read-only file system";
    case errc::resource_deadlock_would_occur:
      return "Resource deadlock avoided";
    case errc::resource_unavailable_try_again:
      return "Resource temporarily unavailable";
    case errc::result_out_of_range:
      return "Numerical result out of range";
    case errc::state_not_recoverable:
      return "State not recoverable";
    case errc::stream_timeout:
      return "Timer expired";
    case errc::text_file_busy:
      return "Text file busy";
    case errc::timed_out:
      return "Connection timed out";
    case errc::too_many_files_open_in_system:
      return "Too many open files in system";
    case errc::too_many_files_open:
      return "Too many open files";
    case errc::too_many_links:
      return "Too many links";
    case errc::too_many_symbolic_link_levels:
      return "Too many levels of symbolic links";
    case errc::value_too_large:
      return "Value too large for defined data type";
    case errc::wrong_protocol_type:
      return "Protocol wrong type for socket";
    default:
      return "unknown";
    }
  }
} // namespace detail
/*! The implementation of the domain for generic status codes, those mapped by `errc` (POSIX).
 */
class _generic_code_domain : public status_code_domain
{
  template <class> friend class status_code;
  template <class StatusCode> friend class detail::indirecting_domain;
  using _base = status_code_domain;
public:
  //! The value type of the generic code, which is an `errc` as per POSIX.
  using value_type = errc;
  using string_ref = _base::string_ref;
public:
  //! Default constructor
  constexpr explicit _generic_code_domain(typename _base::unique_id_type id = 0x746d6354f4f733e9) noexcept
      : _base(id)
  {
  }
  _generic_code_domain(const _generic_code_domain &) = default;
  _generic_code_domain(_generic_code_domain &&) = default;
  _generic_code_domain &operator=(const _generic_code_domain &) = default;
  _generic_code_domain &operator=(_generic_code_domain &&) = default;
  ~_generic_code_domain() = default;
  //! Constexpr singleton getter. Returns the constexpr generic_code_domain variable.
  static inline constexpr const _generic_code_domain &get();
  virtual _base::string_ref name() const noexcept override { return string_ref("generic domain"); } // NOLINT
protected:
  virtual bool _do_failure(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this); // NOLINT
    return static_cast<const generic_code &>(code).value() != errc::success; // NOLINT
  }
  virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept override // NOLINT
  {
    assert(code1.domain() == *this); // NOLINT
    const auto &c1 = static_cast<const generic_code &>(code1); // NOLINT
    if(code2.domain() == *this)
    {
      const auto &c2 = static_cast<const generic_code &>(code2); // NOLINT
      return c1.value() == c2.value();
    }
    return false;
  }
  virtual generic_code _generic_code(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this); // NOLINT
    return static_cast<const generic_code &>(code); // NOLINT
  }
  virtual _base::string_ref _do_message(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this); // NOLINT
    const auto &c = static_cast<const generic_code &>(code); // NOLINT
    return string_ref(detail::generic_code_message(c.value()));
  }
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || 0L
  SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const override // NOLINT
  {
    assert(code.domain() == *this); // NOLINT
    const auto &c = static_cast<const generic_code &>(code); // NOLINT
    throw status_error<_generic_code_domain>(c);
  }
#endif
};
//! A specialisation of `status_error` for the generic code domain.
using generic_error = status_error<_generic_code_domain>;
//! A constexpr source variable for the generic code domain, which is that of `errc` (POSIX). Returned by `_generic_code_domain::get()`.
constexpr _generic_code_domain generic_code_domain;
inline constexpr const _generic_code_domain &_generic_code_domain::get()
{
  return generic_code_domain;
}
// Enable implicit construction of generic_code from errc
SYSTEM_ERROR2_CONSTEXPR14 inline generic_code make_status_code(errc c) noexcept
{
  return generic_code(in_place, c);
}
/*************************************************************************************************************/
template <class T> inline bool status_code<void>::equivalent(const status_code<T> &o) const noexcept
{
  if(_domain && o._domain)
  {
    if(_domain->_do_equivalent(*this, o))
    {
      return true;
    }
    if(o._domain->_do_equivalent(o, *this))
    {
      return true;
    }
    generic_code c1 = o._domain->_generic_code(o);
    if(c1.value() != errc::unknown && _domain->_do_equivalent(*this, c1))
    {
      return true;
    }
    generic_code c2 = _domain->_generic_code(*this);
    if(c2.value() != errc::unknown && o._domain->_do_equivalent(o, c2))
    {
      return true;
    }
  }
  // If we are both empty, we are equivalent, otherwise not equivalent
  return (!_domain && !o._domain);
}
//! True if the status code's are semantically equal via `equivalent()`.
template <class DomainType1, class DomainType2> inline bool operator==(const status_code<DomainType1> &a, const status_code<DomainType2> &b) noexcept
{
  return a.equivalent(b);
}
//! True if the status code's are not semantically equal via `equivalent()`.
template <class DomainType1, class DomainType2> inline bool operator!=(const status_code<DomainType1> &a, const status_code<DomainType2> &b) noexcept
{
  return !a.equivalent(b);
}
//! True if the status code's are semantically equal via `equivalent()` to `make_status_code(T)`.
template <class DomainType1, class T, //
          class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<const T &>::type, // Safe ADL lookup of make_status_code(), returns void if not found
          typename std::enable_if<is_status_code<MakeStatusCodeResult>::value, bool>::type = true> // ADL makes a status code
inline bool operator==(const status_code<DomainType1> &a, const T &b)
{
  return a.equivalent(make_status_code(b));
}
//! True if the status code's are semantically equal via `equivalent()` to `make_status_code(T)`.
template <class T, class DomainType1, //
          class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<const T &>::type, // Safe ADL lookup of make_status_code(), returns void if not found
          typename std::enable_if<is_status_code<MakeStatusCodeResult>::value, bool>::type = true> // ADL makes a status code
inline bool operator==(const T &a, const status_code<DomainType1> &b)
{
  return b.equivalent(make_status_code(a));
}
//! True if the status code's are not semantically equal via `equivalent()` to `make_status_code(T)`.
template <class DomainType1, class T, //
          class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<const T &>::type, // Safe ADL lookup of make_status_code(), returns void if not found
          typename std::enable_if<is_status_code<MakeStatusCodeResult>::value, bool>::type = true> // ADL makes a status code
inline bool operator!=(const status_code<DomainType1> &a, const T &b)
{
  return !a.equivalent(make_status_code(b));
}
//! True if the status code's are semantically equal via `equivalent()` to `make_status_code(T)`.
template <class T, class DomainType1, //
          class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<const T &>::type, // Safe ADL lookup of make_status_code(), returns void if not found
          typename std::enable_if<is_status_code<MakeStatusCodeResult>::value, bool>::type = true> // ADL makes a status code
inline bool operator!=(const T &a, const status_code<DomainType1> &b)
{
  return !b.equivalent(make_status_code(a));
}
//! True if the status code's are semantically equal via `equivalent()` to `quick_status_code_from_enum<T>::code_type(b)`.
template <class DomainType1, class T, //
          class QuickStatusCodeType = typename quick_status_code_from_enum<T>::code_type // Enumeration has been activated
          >
inline bool operator==(const status_code<DomainType1> &a, const T &b)
{
  return a.equivalent(QuickStatusCodeType(b));
}
//! True if the status code's are semantically equal via `equivalent()` to `quick_status_code_from_enum<T>::code_type(a)`.
template <class T, class DomainType1, //
          class QuickStatusCodeType = typename quick_status_code_from_enum<T>::code_type // Enumeration has been activated
          >
inline bool operator==(const T &a, const status_code<DomainType1> &b)
{
  return b.equivalent(QuickStatusCodeType(a));
}
//! True if the status code's are not semantically equal via `equivalent()` to `quick_status_code_from_enum<T>::code_type(b)`.
template <class DomainType1, class T, //
          class QuickStatusCodeType = typename quick_status_code_from_enum<T>::code_type // Enumeration has been activated
          >
inline bool operator!=(const status_code<DomainType1> &a, const T &b)
{
  return !a.equivalent(QuickStatusCodeType(b));
}
//! True if the status code's are not semantically equal via `equivalent()` to `quick_status_code_from_enum<T>::code_type(a)`.
template <class T, class DomainType1, //
          class QuickStatusCodeType = typename quick_status_code_from_enum<T>::code_type // Enumeration has been activated
          >
inline bool operator!=(const T &a, const status_code<DomainType1> &b)
{
  return !b.equivalent(QuickStatusCodeType(a));
}
SYSTEM_ERROR2_NAMESPACE_END
#endif
SYSTEM_ERROR2_NAMESPACE_BEGIN
template <class Enum> class _quick_status_code_from_enum_domain;
//! A status code wrapping `Enum` generated from `quick_status_code_from_enum`.
template <class Enum> using quick_status_code_from_enum_code = status_code<_quick_status_code_from_enum_domain<Enum>>;
//! Defaults for an implementation of `quick_status_code_from_enum<Enum>`
template <class Enum> struct quick_status_code_from_enum_defaults
{
  //! The type of the resulting code
  using code_type = quick_status_code_from_enum_code<Enum>;
  //! Used within `quick_status_code_from_enum` to define a mapping of enumeration value with its status code
  struct mapping
  {
    //! The enumeration type
    using enumeration_type = Enum;
    //! The value being mapped
    const Enum value;
    //! A string representation for this enumeration value
    const char *message;
    //! A list of `errc` equivalents for this enumeration value
    const std::initializer_list<errc> code_mappings;
  };
  //! Used within `quick_status_code_from_enum` to define mixins for the status code wrapping `Enum`
  template <class Base> struct mixin : Base
  {
    using Base::Base;
  };
};
/*! The implementation of the domain for status codes wrapping `Enum` generated from `quick_status_code_from_enum`.
 */
template <class Enum> class _quick_status_code_from_enum_domain : public status_code_domain
{
  template <class DomainType> friend class status_code;
  template <class StatusCode> friend class detail::indirecting_domain;
  using _base = status_code_domain;
  using _src = quick_status_code_from_enum<Enum>;
public:
  //! The value type of the quick status code from enum
  using value_type = Enum;
  using _base::string_ref;
  constexpr _quick_status_code_from_enum_domain()
      : status_code_domain(_src::domain_uuid, _uuid_size<detail::cstrlen(_src::domain_uuid)>())
  {
  }
  _quick_status_code_from_enum_domain(const _quick_status_code_from_enum_domain &) = default;
  _quick_status_code_from_enum_domain(_quick_status_code_from_enum_domain &&) = default;
  _quick_status_code_from_enum_domain &operator=(const _quick_status_code_from_enum_domain &) = default;
  _quick_status_code_from_enum_domain &operator=(_quick_status_code_from_enum_domain &&) = default;
  ~_quick_status_code_from_enum_domain() = default;
#if __cplusplus < 201402L && !defined(_MSC_VER)
  static inline const _quick_status_code_from_enum_domain &get()
  {
    static _quick_status_code_from_enum_domain v;
    return v;
  }
#else
  static inline constexpr const _quick_status_code_from_enum_domain &get();
#endif
  virtual string_ref name() const noexcept override { return string_ref(_src::domain_name); }
protected:
  // Not sure if a hash table is worth it here, most enumerations won't be long enough to be worth it
  // Also, until C++ 20's consteval, the hash table would get emitted into the binary, bloating it
  static SYSTEM_ERROR2_CONSTEXPR14 const typename _src::mapping *_find_mapping(value_type v) noexcept
  {
    for(const auto &i : _src::value_mappings())
    {
      if(i.value == v)
      {
        return &i;
      }
    }
    return nullptr;
  }
  virtual bool _do_failure(const status_code<void> &code) const noexcept override
  {
    assert(code.domain() == *this); // NOLINT
    // If `errc::success` is in the generic code mapping, it is not a failure
    const auto *mapping = _find_mapping(static_cast<const quick_status_code_from_enum_code<value_type> &>(code).value());
    assert(mapping != nullptr);
    if(mapping != nullptr)
    {
      for(errc ec : mapping->code_mappings)
      {
        if(ec == errc::success)
        {
          return false;
        }
      }
    }
    return true;
  }
  virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept override
  {
    assert(code1.domain() == *this); // NOLINT
    const auto &c1 = static_cast<const quick_status_code_from_enum_code<value_type> &>(code1); // NOLINT
    if(code2.domain() == *this)
    {
      const auto &c2 = static_cast<const quick_status_code_from_enum_code<value_type> &>(code2); // NOLINT
      return c1.value() == c2.value();
    }
    if(code2.domain() == generic_code_domain)
    {
      const auto &c2 = static_cast<const generic_code &>(code2); // NOLINT
      const auto *mapping = _find_mapping(c1.value());
      assert(mapping != nullptr);
      if(mapping != nullptr)
      {
        for(errc ec : mapping->code_mappings)
        {
          if(ec == c2.value())
          {
            return true;
          }
        }
      }
    }
    return false;
  }
  virtual generic_code _generic_code(const status_code<void> &code) const noexcept override
  {
    assert(code.domain() == *this); // NOLINT
    const auto *mapping = _find_mapping(static_cast<const quick_status_code_from_enum_code<value_type> &>(code).value());
    assert(mapping != nullptr);
    if(mapping != nullptr)
    {
      if(mapping->code_mappings.size() > 0)
      {
        return *mapping->code_mappings.begin();
      }
    }
    return errc::unknown;
  }
  virtual string_ref _do_message(const status_code<void> &code) const noexcept override
  {
    assert(code.domain() == *this); // NOLINT
    const auto *mapping = _find_mapping(static_cast<const quick_status_code_from_enum_code<value_type> &>(code).value());
    assert(mapping != nullptr);
    if(mapping != nullptr)
    {
      return string_ref(mapping->message);
    }
    return string_ref("unknown");
  }
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || 0L
  SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const override
  {
    assert(code.domain() == *this); // NOLINT
    const auto &c = static_cast<const quick_status_code_from_enum_code<value_type> &>(code); // NOLINT
    throw status_error<_quick_status_code_from_enum_domain>(c);
  }
#endif
};
#if __cplusplus >= 201402L || defined(_MSC_VER)
template <class Enum> constexpr _quick_status_code_from_enum_domain<Enum> quick_status_code_from_enum_domain = {};
template <class Enum> inline constexpr const _quick_status_code_from_enum_domain<Enum> &_quick_status_code_from_enum_domain<Enum>::get()
{
  return quick_status_code_from_enum_domain<Enum>;
}
#endif
namespace mixins
{
  template <class Base, class Enum> struct mixin<Base, _quick_status_code_from_enum_domain<Enum>> : public quick_status_code_from_enum<Enum>::template mixin<Base>
  {
    using quick_status_code_from_enum<Enum>::template mixin<Base>::mixin;
  };
} // namespace mixins
SYSTEM_ERROR2_NAMESPACE_END
#endif
/* Pointer to a SG14 status_code
(C) 2018 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Sep 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_STATUS_CODE_PTR_HPP
#define SYSTEM_ERROR2_STATUS_CODE_PTR_HPP
SYSTEM_ERROR2_NAMESPACE_BEGIN
namespace detail
{
  template <class StatusCode> class indirecting_domain : public status_code_domain
  {
    template <class DomainType> friend class status_code;
    using _base = status_code_domain;
  public:
    using value_type = StatusCode *;
    using _base::string_ref;
    constexpr indirecting_domain() noexcept
        : _base(0xc44f7bdeb2cc50e9 ^ typename StatusCode::domain_type().id() /* unique-ish based on domain's unique id */)
    {
    }
    indirecting_domain(const indirecting_domain &) = default;
    indirecting_domain(indirecting_domain &&) = default; // NOLINT
    indirecting_domain &operator=(const indirecting_domain &) = default;
    indirecting_domain &operator=(indirecting_domain &&) = default; // NOLINT
    ~indirecting_domain() = default;
#if __cplusplus < 201402L && !defined(_MSC_VER)
    static inline const indirecting_domain &get()
    {
      static indirecting_domain v;
      return v;
    }
#else
    static inline constexpr const indirecting_domain &get();
#endif
    virtual string_ref name() const noexcept override { return typename StatusCode::domain_type().name(); } // NOLINT
  protected:
    using _mycode = status_code<indirecting_domain>;
    virtual bool _do_failure(const status_code<void> &code) const noexcept override // NOLINT
    {
      assert(code.domain() == *this);
      const auto &c = static_cast<const _mycode &>(code); // NOLINT
      return typename StatusCode::domain_type()._do_failure(*c.value());
    }
    virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept override // NOLINT
    {
      assert(code1.domain() == *this);
      const auto &c1 = static_cast<const _mycode &>(code1); // NOLINT
      return typename StatusCode::domain_type()._do_equivalent(*c1.value(), code2);
    }
    virtual generic_code _generic_code(const status_code<void> &code) const noexcept override // NOLINT
    {
      assert(code.domain() == *this);
      const auto &c = static_cast<const _mycode &>(code); // NOLINT
      return typename StatusCode::domain_type()._generic_code(*c.value());
    }
    virtual string_ref _do_message(const status_code<void> &code) const noexcept override // NOLINT
    {
      assert(code.domain() == *this);
      const auto &c = static_cast<const _mycode &>(code); // NOLINT
      return typename StatusCode::domain_type()._do_message(*c.value());
    }
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || 0L
    SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const override // NOLINT
    {
      assert(code.domain() == *this);
      const auto &c = static_cast<const _mycode &>(code); // NOLINT
      typename StatusCode::domain_type()._do_throw_exception(*c.value());
      std::abort();
    }
#endif
    virtual void _do_erased_copy(status_code<void> &dst, const status_code<void> &src, size_t /*unused*/) const override // NOLINT
    {
      // Note that dst will not have its domain set
      assert(src.domain() == *this);
      auto &d = static_cast<_mycode &>(dst); // NOLINT
      const auto &_s = static_cast<const _mycode &>(src); // NOLINT
      const StatusCode &s = *_s.value();
      new(&d) _mycode(in_place, new StatusCode(s));
    }
    virtual void _do_erased_destroy(status_code<void> &code, size_t /*unused*/) const noexcept override // NOLINT
    {
      assert(code.domain() == *this);
      auto &c = static_cast<_mycode &>(code); // NOLINT
      delete c.value(); // NOLINT
    }
  };
#if __cplusplus >= 201402L || defined(_MSC_VER)
  template <class StatusCode> constexpr indirecting_domain<StatusCode> _indirecting_domain{};
  template <class StatusCode> inline constexpr const indirecting_domain<StatusCode> &indirecting_domain<StatusCode>::get() { return _indirecting_domain<StatusCode>; }
#endif
} // namespace detail
/*! Make an erased status code which indirects to a dynamically allocated status code.
This is useful for shoehorning a rich status code with large value type into a small
erased status code like `system_code`, with which the status code generated by this
function is compatible. Note that this function can throw due to `bad_alloc`.
*/
template <class T, typename std::enable_if<is_status_code<T>::value, bool>::type = true> //
inline status_code<erased<typename std::add_pointer<typename std::decay<T>::type>::type>> make_status_code_ptr(T &&v)
{
  using status_code_type = typename std::decay<T>::type;
  return status_code<detail::indirecting_domain<status_code_type>>(in_place, new status_code_type(static_cast<T &&>(v)));
}
/*! If a status code refers to a `status_code_ptr` which indirects to a status
code of type `StatusCode`, return a pointer to that `StatusCode`. Otherwise return null.
*/
template <class StatusCode, class U, typename std::enable_if<is_status_code<StatusCode>::value, bool>::type = true> inline StatusCode *get_if(status_code<erased<U>> *v) noexcept
{
  if((0xc44f7bdeb2cc50e9 ^ typename StatusCode::domain_type().id()) != v->domain().id())
  {
    return nullptr;
  }
  union {
    U value;
    StatusCode *ret;
  };
  value = v->value();
  return ret;
}
//! \overload Const overload
template <class StatusCode, class U, typename std::enable_if<is_status_code<StatusCode>::value, bool>::type = true> inline const StatusCode *get_if(const status_code<erased<U>> *v) noexcept
{
  if((0xc44f7bdeb2cc50e9 ^ typename StatusCode::domain_type().id()) != v->domain().id())
  {
    return nullptr;
  }
  union {
    U value;
    const StatusCode *ret;
  };
  value = v->value();
  return ret;
}
/*! If a status code refers to a `status_code_ptr`, return the id of the erased
status code's domain. Otherwise return a meaningless number.
*/
template <class U> inline typename status_code_domain::unique_id_type get_id(const status_code<erased<U>> &v) noexcept
{
  return 0xc44f7bdeb2cc50e9 ^ v.domain().id();
}
SYSTEM_ERROR2_NAMESPACE_END
#endif
SYSTEM_ERROR2_NAMESPACE_BEGIN
/*! A `status_code` which is always a failure. The closest equivalent to
`std::error_code`, except it cannot be modified, and is templated.

Differences from `status_code`:

- Never successful (this contract is checked on construction, if fails then it
terminates the process).
- Is immutable.
*/
template <class DomainType> class errored_status_code : public status_code<DomainType>
{
  using _base = status_code<DomainType>;
  using _base::clear;
  using _base::success;
  void _check()
  {
    if(_base::success())
    {
      std::terminate();
    }
  }
public:
  //! The type of the erased error code.
  using typename _base::value_type;
  //! The type of a reference to a message string.
  using typename _base::string_ref;
  //! Default constructor.
  errored_status_code() = default;
  //! Copy constructor.
  errored_status_code(const errored_status_code &) = default;
  //! Move constructor.
  errored_status_code(errored_status_code &&) = default; // NOLINT
  //! Copy assignment.
  errored_status_code &operator=(const errored_status_code &) = default;
  //! Move assignment.
  errored_status_code &operator=(errored_status_code &&) = default; // NOLINT
  ~errored_status_code() = default;
  //! Explicitly construct from any similarly erased status code
  explicit errored_status_code(const _base &o) noexcept(std::is_nothrow_copy_constructible<_base>::value)
      : _base(o)
  {
    _check();
  }
  //! Explicitly construct from any similarly erased status code
  explicit errored_status_code(_base &&o) noexcept(std::is_nothrow_move_constructible<_base>::value)
      : _base(static_cast<_base &&>(o))
  {
    _check();
  }
  /***** KEEP THESE IN SYNC WITH STATUS_CODE *****/
  //! Implicit construction from any type where an ADL discovered `make_status_code(T, Args ...)` returns a `status_code`.
  template <class T, class... Args, //
            class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<T, Args...>::type, // Safe ADL lookup of make_status_code(), returns void if not found
            typename std::enable_if<!std::is_same<typename std::decay<T>::type, errored_status_code>::value // not copy/move of self
                                    && !std::is_same<typename std::decay<T>::type, in_place_t>::value // not in_place_t
                                    && is_status_code<MakeStatusCodeResult>::value // ADL makes a status code
                                    && std::is_constructible<errored_status_code, MakeStatusCodeResult>::value, // ADLed status code is compatible
                                    bool>::type = true>
  errored_status_code(T &&v, Args &&... args) noexcept(noexcept(make_status_code(std::declval<T>(), std::declval<Args>()...))) // NOLINT
      : errored_status_code(make_status_code(static_cast<T &&>(v), static_cast<Args &&>(args)...))
  {
    _check();
  }
  //! Implicit construction from any `quick_status_code_from_enum<Enum>` enumerated type.
  template <class Enum, //
            class QuickStatusCodeType = typename quick_status_code_from_enum<Enum>::code_type, // Enumeration has been activated
            typename std::enable_if<std::is_constructible<errored_status_code, QuickStatusCodeType>::value, // Its status code is compatible
                                    bool>::type = true>
  errored_status_code(Enum &&v) noexcept(std::is_nothrow_constructible<errored_status_code, QuickStatusCodeType>::value) // NOLINT
      : errored_status_code(QuickStatusCodeType(static_cast<Enum &&>(v)))
  {
    _check();
  }
  //! Explicit in-place construction.
  template <class... Args>
  explicit errored_status_code(in_place_t _, Args &&... args) noexcept(std::is_nothrow_constructible<value_type, Args &&...>::value)
      : _base(_, static_cast<Args &&>(args)...)
  {
    _check();
  }
  //! Explicit in-place construction from initialiser list.
  template <class T, class... Args>
  explicit errored_status_code(in_place_t _, std::initializer_list<T> il, Args &&... args) noexcept(std::is_nothrow_constructible<value_type, std::initializer_list<T>, Args &&...>::value)
      : _base(_, il, static_cast<Args &&>(args)...)
  {
    _check();
  }
  //! Explicit copy construction from a `value_type`.
  explicit errored_status_code(const value_type &v) noexcept(std::is_nothrow_copy_constructible<value_type>::value)
      : _base(v)
  {
    _check();
  }
  //! Explicit move construction from a `value_type`.
  explicit errored_status_code(value_type &&v) noexcept(std::is_nothrow_move_constructible<value_type>::value)
      : _base(static_cast<value_type &&>(v))
  {
    _check();
  }
  /*! Explicit construction from an erased status code. Available only if
  `value_type` is trivially destructible and `sizeof(status_code) <= sizeof(status_code<erased<>>)`.
  Does not check if domains are equal.
  */
  template <class ErasedType, //
            typename std::enable_if<detail::type_erasure_is_safe<ErasedType, value_type>::value, bool>::type = true>
  explicit errored_status_code(const status_code<erased<ErasedType>> &v) noexcept(std::is_nothrow_copy_constructible<value_type>::value)
      : errored_status_code(detail::erasure_cast<value_type>(v.value())) // NOLINT
  {
    assert(v.domain() == this->domain()); // NOLINT
    _check();
  }
  //! Always false (including at compile time), as errored status codes are never successful.
  constexpr bool success() const noexcept { return false; }
  //! Return a const reference to the `value_type`.
  constexpr const value_type &value() const &noexcept { return this->_value; }
};
namespace traits
{
  template <class DomainType> struct is_move_bitcopying<errored_status_code<DomainType>>
  {
    static constexpr bool value = is_move_bitcopying<typename DomainType::value_type>::value;
  };
} // namespace traits
template <class ErasedType> class errored_status_code<erased<ErasedType>> : public status_code<erased<ErasedType>>
{
  using _base = status_code<erased<ErasedType>>;
  using _base::success;
  void _check()
  {
    if(_base::success())
    {
      std::terminate();
    }
  }
public:
  using value_type = typename _base::value_type;
  using string_ref = typename _base::string_ref;
  //! Default construction to empty
  errored_status_code() = default;
  //! Copy constructor
  errored_status_code(const errored_status_code &) = default;
  //! Move constructor
  errored_status_code(errored_status_code &&) = default; // NOLINT
                                                          //! Copy assignment
  errored_status_code &operator=(const errored_status_code &) = default;
  //! Move assignment
  errored_status_code &operator=(errored_status_code &&) = default; // NOLINT
  ~errored_status_code() = default;
  //! Explicitly construct from any similarly erased status code
  explicit errored_status_code(const _base &o) noexcept(std::is_nothrow_copy_constructible<_base>::value)
      : _base(o)
  {
    _check();
  }
  //! Explicitly construct from any similarly erased status code
  explicit errored_status_code(_base &&o) noexcept(std::is_nothrow_move_constructible<_base>::value)
      : _base(static_cast<_base &&>(o))
  {
    _check();
  }
  /***** KEEP THESE IN SYNC WITH STATUS_CODE *****/
  //! Implicit copy construction from any other status code if its value type is trivially copyable and it would fit into our storage
  template <class DomainType, //
            typename std::enable_if<std::is_trivially_copyable<typename DomainType::value_type>::value //
                                    && detail::type_erasure_is_safe<value_type, typename DomainType::value_type>::value,
                                    bool>::type = true>
  errored_status_code(const status_code<DomainType> &v) noexcept
      : _base(v) // NOLINT
  {
    _check();
  }
  //! Implicit copy construction from any other status code if its value type is trivially copyable and it would fit into our storage
  template <class DomainType, //
            typename std::enable_if<std::is_trivially_copyable<typename DomainType::value_type>::value //
                                    && detail::type_erasure_is_safe<value_type, typename DomainType::value_type>::value,
                                    bool>::type = true>
  errored_status_code(const errored_status_code<DomainType> &v) noexcept
      : _base(static_cast<const status_code<DomainType> &>(v)) // NOLINT
  {
    _check();
  }
  //! Implicit move construction from any other status code if its value type is trivially copyable or move bitcopying and it would fit into our storage
  template <class DomainType, //
            typename std::enable_if<detail::type_erasure_is_safe<value_type, typename DomainType::value_type>::value,
                                    bool>::type = true>
  errored_status_code(status_code<DomainType> &&v) noexcept
      : _base(static_cast<status_code<DomainType> &&>(v)) // NOLINT
  {
    _check();
  }
  //! Implicit move construction from any other status code if its value type is trivially copyable or move bitcopying and it would fit into our storage
  template <class DomainType, //
            typename std::enable_if<detail::type_erasure_is_safe<value_type, typename DomainType::value_type>::value,
                                    bool>::type = true>
  errored_status_code(errored_status_code<DomainType> &&v) noexcept
      : _base(static_cast<status_code<DomainType> &&>(v)) // NOLINT
  {
    _check();
  }
  //! Implicit construction from any type where an ADL discovered `make_status_code(T, Args ...)` returns a `status_code`.
  template <class T, class... Args, //
            class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<T, Args...>::type, // Safe ADL lookup of make_status_code(), returns void if not found
            typename std::enable_if<!std::is_same<typename std::decay<T>::type, errored_status_code>::value // not copy/move of self
                                    && !std::is_same<typename std::decay<T>::type, value_type>::value // not copy/move of value type
                                    && is_status_code<MakeStatusCodeResult>::value // ADL makes a status code
                                    && std::is_constructible<errored_status_code, MakeStatusCodeResult>::value, // ADLed status code is compatible
                                    bool>::type = true>
  errored_status_code(T &&v, Args &&... args) noexcept(noexcept(make_status_code(std::declval<T>(), std::declval<Args>()...))) // NOLINT
      : errored_status_code(make_status_code(static_cast<T &&>(v), static_cast<Args &&>(args)...))
  {
    _check();
  }
  //! Implicit construction from any `quick_status_code_from_enum<Enum>` enumerated type.
  template <class Enum, //
            class QuickStatusCodeType = typename quick_status_code_from_enum<Enum>::code_type, // Enumeration has been activated
            typename std::enable_if<std::is_constructible<errored_status_code, QuickStatusCodeType>::value, // Its status code is compatible
                                    bool>::type = true>
  errored_status_code(Enum &&v) noexcept(std::is_nothrow_constructible<errored_status_code, QuickStatusCodeType>::value) // NOLINT
      : errored_status_code(QuickStatusCodeType(static_cast<Enum &&>(v)))
  {
    _check();
  }
  //! Always false (including at compile time), as errored status codes are never successful.
  constexpr bool success() const noexcept { return false; }
  //! Return the erased `value_type` by value.
  constexpr value_type value() const noexcept { return this->_value; }
};
namespace traits
{
  template <class ErasedType> struct is_move_bitcopying<errored_status_code<erased<ErasedType>>>
  {
    static constexpr bool value = true;
  };
} // namespace traits
//! True if the status code's are semantically equal via `equivalent()`.
template <class DomainType1, class DomainType2> inline bool operator==(const errored_status_code<DomainType1> &a, const errored_status_code<DomainType2> &b) noexcept
{
  return a.equivalent(static_cast<const status_code<DomainType2> &>(b));
}
//! True if the status code's are semantically equal via `equivalent()`.
template <class DomainType1, class DomainType2> inline bool operator==(const status_code<DomainType1> &a, const errored_status_code<DomainType2> &b) noexcept
{
  return a.equivalent(static_cast<const status_code<DomainType2> &>(b));
}
//! True if the status code's are semantically equal via `equivalent()`.
template <class DomainType1, class DomainType2> inline bool operator==(const errored_status_code<DomainType1> &a, const status_code<DomainType2> &b) noexcept
{
  return static_cast<const status_code<DomainType1> &>(a).equivalent(b);
}
//! True if the status code's are not semantically equal via `equivalent()`.
template <class DomainType1, class DomainType2> inline bool operator!=(const errored_status_code<DomainType1> &a, const errored_status_code<DomainType2> &b) noexcept
{
  return !a.equivalent(static_cast<const status_code<DomainType2> &>(b));
}
//! True if the status code's are not semantically equal via `equivalent()`.
template <class DomainType1, class DomainType2> inline bool operator!=(const status_code<DomainType1> &a, const errored_status_code<DomainType2> &b) noexcept
{
  return !a.equivalent(static_cast<const status_code<DomainType2> &>(b));
}
//! True if the status code's are not semantically equal via `equivalent()`.
template <class DomainType1, class DomainType2> inline bool operator!=(const errored_status_code<DomainType1> &a, const status_code<DomainType2> &b) noexcept
{
  return !static_cast<const status_code<DomainType1> &>(a).equivalent(b);
}
//! True if the status code's are semantically equal via `equivalent()` to `make_status_code(T)`.
template <class DomainType1, class T, //
          class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<const T &>::type, // Safe ADL lookup of make_status_code(), returns void if not found
          typename std::enable_if<is_status_code<MakeStatusCodeResult>::value, bool>::type = true> // ADL makes a status code
inline bool operator==(const errored_status_code<DomainType1> &a, const T &b)
{
  return a.equivalent(make_status_code(b));
}
//! True if the status code's are semantically equal via `equivalent()` to `make_status_code(T)`.
template <class T, class DomainType1, //
          class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<const T &>::type, // Safe ADL lookup of make_status_code(), returns void if not found
          typename std::enable_if<is_status_code<MakeStatusCodeResult>::value, bool>::type = true> // ADL makes a status code
inline bool operator==(const T &a, const errored_status_code<DomainType1> &b)
{
  return b.equivalent(make_status_code(a));
}
//! True if the status code's are not semantically equal via `equivalent()` to `make_status_code(T)`.
template <class DomainType1, class T, //
          class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<const T &>::type, // Safe ADL lookup of make_status_code(), returns void if not found
          typename std::enable_if<is_status_code<MakeStatusCodeResult>::value, bool>::type = true> // ADL makes a status code
inline bool operator!=(const errored_status_code<DomainType1> &a, const T &b)
{
  return !a.equivalent(make_status_code(b));
}
//! True if the status code's are semantically equal via `equivalent()` to `make_status_code(T)`.
template <class T, class DomainType1, //
          class MakeStatusCodeResult = typename detail::safe_get_make_status_code_result<const T &>::type, // Safe ADL lookup of make_status_code(), returns void if not found
          typename std::enable_if<is_status_code<MakeStatusCodeResult>::value, bool>::type = true> // ADL makes a status code
inline bool operator!=(const T &a, const errored_status_code<DomainType1> &b)
{
  return !b.equivalent(make_status_code(a));
}
//! True if the status code's are semantically equal via `equivalent()` to `quick_status_code_from_enum<T>::code_type(b)`.
template <class DomainType1, class T, //
          class QuickStatusCodeType = typename quick_status_code_from_enum<T>::code_type // Enumeration has been activated
          >
inline bool operator==(const errored_status_code<DomainType1> &a, const T &b)
{
  return a.equivalent(QuickStatusCodeType(b));
}
//! True if the status code's are semantically equal via `equivalent()` to `quick_status_code_from_enum<T>::code_type(a)`.
template <class T, class DomainType1, //
          class QuickStatusCodeType = typename quick_status_code_from_enum<T>::code_type // Enumeration has been activated
          >
inline bool operator==(const T &a, const errored_status_code<DomainType1> &b)
{
  return b.equivalent(QuickStatusCodeType(a));
}
//! True if the status code's are not semantically equal via `equivalent()` to `quick_status_code_from_enum<T>::code_type(b)`.
template <class DomainType1, class T, //
          class QuickStatusCodeType = typename quick_status_code_from_enum<T>::code_type // Enumeration has been activated
          >
inline bool operator!=(const errored_status_code<DomainType1> &a, const T &b)
{
  return !a.equivalent(QuickStatusCodeType(b));
}
//! True if the status code's are not semantically equal via `equivalent()` to `quick_status_code_from_enum<T>::code_type(a)`.
template <class T, class DomainType1, //
          class QuickStatusCodeType = typename quick_status_code_from_enum<T>::code_type // Enumeration has been activated
          >
inline bool operator!=(const T &a, const errored_status_code<DomainType1> &b)
{
  return !b.equivalent(QuickStatusCodeType(a));
}
namespace detail
{
  template <class T> struct is_errored_status_code
  {
    static constexpr bool value = false;
  };
  template <class T> struct is_errored_status_code<errored_status_code<T>>
  {
    static constexpr bool value = true;
  };
  template <class T> struct is_erased_errored_status_code
  {
    static constexpr bool value = false;
  };
  template <class T> struct is_erased_errored_status_code<errored_status_code<erased<T>>>
  {
    static constexpr bool value = true;
  };
} // namespace detail
//! Trait returning true if the type is an errored status code.
template <class T> struct is_errored_status_code
{
  static constexpr bool value = detail::is_errored_status_code<typename std::decay<T>::type>::value || detail::is_erased_errored_status_code<typename std::decay<T>::type>::value;
};
SYSTEM_ERROR2_NAMESPACE_END
#endif
/* Proposed SG14 status_code
(C) 2018 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_SYSTEM_CODE_HPP
#define SYSTEM_ERROR2_SYSTEM_CODE_HPP
#ifndef SYSTEM_ERROR2_NOT_POSIX
/* Proposed SG14 status_code
(C) 2018-2020 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_POSIX_CODE_HPP
#define SYSTEM_ERROR2_POSIX_CODE_HPP
#ifdef SYSTEM_ERROR2_NOT_POSIX
#error <posix_code.hpp> is not includable when SYSTEM_ERROR2_NOT_POSIX is defined!
#endif
#include <cstring> // for strchr and strerror_r
SYSTEM_ERROR2_NAMESPACE_BEGIN
class _posix_code_domain;
//! A POSIX error code, those returned by `errno`.
using posix_code = status_code<_posix_code_domain>;
//! A specialisation of `status_error` for the POSIX error code domain.
using posix_error = status_error<_posix_code_domain>;
namespace mixins
{
  template <class Base> struct mixin<Base, _posix_code_domain> : public Base
  {
    using Base::Base;
    //! Returns a `posix_code` for the current value of `errno`.
    static posix_code current() noexcept;
  };
} // namespace mixins
/*! The implementation of the domain for POSIX error codes, those returned by `errno`.
 */
class _posix_code_domain : public status_code_domain
{
  template <class DomainType> friend class status_code;
  template <class StatusCode> friend class detail::indirecting_domain;
  using _base = status_code_domain;
  static _base::string_ref _make_string_ref(int c) noexcept
  {
    char buffer[1024] = "";
#ifdef _WIN32
    strerror_s(buffer, sizeof(buffer), c);
#elif defined(__gnu_linux__) && !defined(__ANDROID__) // handle glibc's weird strerror_r()
    char *s = strerror_r(c, buffer, sizeof(buffer)); // NOLINT
    if(s != nullptr)
    {
      strncpy(buffer, s, sizeof(buffer)); // NOLINT
      buffer[1023] = 0;
    }
#else
    strerror_r(c, buffer, sizeof(buffer));
#endif
    size_t length = strlen(buffer); // NOLINT
    auto *p = static_cast<char *>(malloc(length + 1)); // NOLINT
    if(p == nullptr)
    {
      return _base::string_ref("failed to get message from system");
    }
    memcpy(p, buffer, length + 1); // NOLINT
    return _base::atomic_refcounted_string_ref(p, length);
  }
public:
  //! The value type of the POSIX code, which is an `int`
  using value_type = int;
  using _base::string_ref;
  //! Default constructor
  constexpr explicit _posix_code_domain(typename _base::unique_id_type id = 0xa59a56fe5f310933) noexcept
      : _base(id)
  {
  }
  _posix_code_domain(const _posix_code_domain &) = default;
  _posix_code_domain(_posix_code_domain &&) = default;
  _posix_code_domain &operator=(const _posix_code_domain &) = default;
  _posix_code_domain &operator=(_posix_code_domain &&) = default;
  ~_posix_code_domain() = default;
  //! Constexpr singleton getter. Returns constexpr posix_code_domain variable.
  static inline constexpr const _posix_code_domain &get();
  virtual string_ref name() const noexcept override { return string_ref("posix domain"); } // NOLINT
protected:
  virtual bool _do_failure(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this); // NOLINT
    return static_cast<const posix_code &>(code).value() != 0; // NOLINT
  }
  virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept override // NOLINT
  {
    assert(code1.domain() == *this); // NOLINT
    const auto &c1 = static_cast<const posix_code &>(code1); // NOLINT
    if(code2.domain() == *this)
    {
      const auto &c2 = static_cast<const posix_code &>(code2); // NOLINT
      return c1.value() == c2.value();
    }
    if(code2.domain() == generic_code_domain)
    {
      const auto &c2 = static_cast<const generic_code &>(code2); // NOLINT
      if(static_cast<int>(c2.value()) == c1.value())
      {
        return true;
      }
    }
    return false;
  }
  virtual generic_code _generic_code(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this); // NOLINT
    const auto &c = static_cast<const posix_code &>(code); // NOLINT
    return generic_code(static_cast<errc>(c.value()));
  }
  virtual string_ref _do_message(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this); // NOLINT
    const auto &c = static_cast<const posix_code &>(code); // NOLINT
    return _make_string_ref(c.value());
  }
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || 0L
  SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const override // NOLINT
  {
    assert(code.domain() == *this); // NOLINT
    const auto &c = static_cast<const posix_code &>(code); // NOLINT
    throw status_error<_posix_code_domain>(c);
  }
#endif
};
//! A constexpr source variable for the POSIX code domain, which is that of `errno`. Returned by `_posix_code_domain::get()`.
constexpr _posix_code_domain posix_code_domain;
inline constexpr const _posix_code_domain &_posix_code_domain::get()
{
  return posix_code_domain;
}
namespace mixins
{
  template <class Base> inline posix_code mixin<Base, _posix_code_domain>::current() noexcept { return posix_code(errno); }
} // namespace mixins
SYSTEM_ERROR2_NAMESPACE_END
#endif
#else
#endif
#if defined(_WIN32) || 0L
/* Proposed SG14 status_code
(C) 2018 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_NT_CODE_HPP
#define SYSTEM_ERROR2_NT_CODE_HPP
#if !defined(_WIN32) && !0L
#error This file should only be included on Windows
#endif
/* Proposed SG14 status_code
(C) 2018 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Feb 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
(See accompanying file Licence.txt or copy at
http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef SYSTEM_ERROR2_WIN32_CODE_HPP
#define SYSTEM_ERROR2_WIN32_CODE_HPP
#if !defined(_WIN32) && !0L
#error This file should only be included on Windows
#endif
SYSTEM_ERROR2_NAMESPACE_BEGIN
//! \exclude
namespace win32
{
  // A Win32 DWORD
  using DWORD = unsigned long;
  // Used to retrieve the current Win32 error code
  extern DWORD __stdcall GetLastError();
  // Used to retrieve a locale-specific message string for some error code
  extern DWORD __stdcall FormatMessageW(DWORD dwFlags, const void *lpSource, DWORD dwMessageId, DWORD dwLanguageId, wchar_t *lpBuffer, DWORD nSize, void /*va_list*/ *Arguments);
  // Converts UTF-16 message string to UTF-8
  extern int __stdcall WideCharToMultiByte(unsigned int CodePage, DWORD dwFlags, const wchar_t *lpWideCharStr, int cchWideChar, char *lpMultiByteStr, int cbMultiByte, const char *lpDefaultChar, int *lpUsedDefaultChar);
#pragma comment(lib, "kernel32.lib")
#if (defined(__x86_64__) || defined(_M_X64)) || (defined(__aarch64__) || defined(_M_ARM64))
#pragma comment(linker, "/alternatename:?GetLastError@win32@system_error2@@YAKXZ=GetLastError")
#pragma comment(linker, "/alternatename:?FormatMessageW@win32@system_error2@@YAKKPEBXKKPEA_WKPEAX@Z=FormatMessageW")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@system_error2@@YAHIKPEB_WHPEADHPEBDPEAH@Z=WideCharToMultiByte")
#elif defined(__x86__) || defined(_M_IX86) || defined(__i386__)
#pragma comment(linker, "/alternatename:?GetLastError@win32@system_error2@@YGKXZ=__imp__GetLastError@0")
#pragma comment(linker, "/alternatename:?FormatMessageW@win32@system_error2@@YGKKPBXKKPA_WKPAX@Z=__imp__FormatMessageW@28")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@system_error2@@YGHIKPB_WHPADHPBDPAH@Z=__imp__WideCharToMultiByte@32")
#elif defined(__arm__) || defined(_M_ARM)
#pragma comment(linker, "/alternatename:?GetLastError@win32@system_error2@@YAKXZ=GetLastError")
#pragma comment(linker, "/alternatename:?FormatMessageW@win32@system_error2@@YAKKPBXKKPA_WKPAX@Z=FormatMessageW")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@system_error2@@YAHIKPB_WHPADHPBDPAH@Z=WideCharToMultiByte")
#else
#error Unknown architecture
#endif
} // namespace win32
class _win32_code_domain;
class _com_code_domain;
//! (Windows only) A Win32 error code, those returned by `GetLastError()`.
using win32_code = status_code<_win32_code_domain>;
//! (Windows only) A specialisation of `status_error` for the Win32 error code domain.
using win32_error = status_error<_win32_code_domain>;
namespace mixins
{
  template <class Base> struct mixin<Base, _win32_code_domain> : public Base
  {
    using Base::Base;
    //! Returns a `win32_code` for the current value of `GetLastError()`.
    static inline win32_code current() noexcept;
  };
} // namespace mixins
/*! (Windows only) The implementation of the domain for Win32 error codes, those returned by `GetLastError()`.
 */
class _win32_code_domain : public status_code_domain
{
  template <class DomainType> friend class status_code;
  template <class StatusCode> friend class detail::indirecting_domain;
  friend class _com_code_domain;
  using _base = status_code_domain;
  static int _win32_code_to_errno(win32::DWORD c)
  {
    switch(c)
    {
    case 0:
      return 0;
case 0x1: return ENOSYS;
case 0x2: return ENOENT;
case 0x3: return ENOENT;
case 0x4: return EMFILE;
case 0x5: return EACCES;
case 0x6: return EINVAL;
case 0x8: return ENOMEM;
case 0xc: return EACCES;
case 0xe: return ENOMEM;
case 0xf: return ENODEV;
case 0x10: return EACCES;
case 0x11: return EXDEV;
case 0x13: return EACCES;
case 0x14: return ENODEV;
case 0x15: return EAGAIN;
case 0x19: return EIO;
case 0x1d: return EIO;
case 0x1e: return EIO;
case 0x20: return EACCES;
case 0x21: return ENOLCK;
case 0x27: return ENOSPC;
case 0x37: return ENODEV;
case 0x50: return EEXIST;
case 0x52: return EACCES;
case 0x57: return EINVAL;
case 0x6e: return EIO;
case 0x6f: return ENAMETOOLONG;
case 0x70: return ENOSPC;
case 0x7b: return EINVAL;
case 0x83: return EINVAL;
case 0x8e: return EBUSY;
case 0x91: return ENOTEMPTY;
case 0xaa: return EBUSY;
case 0xb7: return EEXIST;
case 0xd4: return ENOLCK;
case 0x10b: return EINVAL;
case 0x3e3: return ECANCELED;
case 0x3e6: return EACCES;
case 0x3f3: return EIO;
case 0x3f4: return EIO;
case 0x3f5: return EIO;
case 0x4d5: return EAGAIN;
case 0x961: return EBUSY;
case 0x964: return EBUSY;
case 0x2714: return EINTR;
case 0x2719: return EBADF;
case 0x271d: return EACCES;
case 0x271e: return EFAULT;
case 0x2726: return EINVAL;
case 0x2728: return EMFILE;
case 0x2733: return EWOULDBLOCK;
case 0x2734: return EINPROGRESS;
case 0x2735: return EALREADY;
case 0x2736: return ENOTSOCK;
case 0x2737: return EDESTADDRREQ;
case 0x2738: return EMSGSIZE;
case 0x2739: return EPROTOTYPE;
case 0x273a: return ENOPROTOOPT;
case 0x273b: return EPROTONOSUPPORT;
case 0x273d: return EOPNOTSUPP;
case 0x273f: return EAFNOSUPPORT;
case 0x2740: return EADDRINUSE;
case 0x2741: return EADDRNOTAVAIL;
case 0x2742: return ENETDOWN;
case 0x2743: return ENETUNREACH;
case 0x2744: return ENETRESET;
case 0x2745: return ECONNABORTED;
case 0x2746: return ECONNRESET;
case 0x2747: return ENOBUFS;
case 0x2748: return EISCONN;
case 0x2749: return ENOTCONN;
case 0x274c: return ETIMEDOUT;
case 0x274d: return ECONNREFUSED;
case 0x274f: return ENAMETOOLONG;
case 0x2751: return EHOSTUNREACH;
    }
    return -1;
  }
  //! Construct from a Win32 error code
  static _base::string_ref _make_string_ref(win32::DWORD c) noexcept
  {
    wchar_t buffer[32768];
    win32::DWORD wlen = win32::FormatMessageW(0x00001000 /*FORMAT_MESSAGE_FROM_SYSTEM*/ | 0x00000200 /*FORMAT_MESSAGE_IGNORE_INSERTS*/, nullptr, c, 0, buffer, 32768, nullptr);
    size_t allocation = wlen + (wlen >> 1);
    win32::DWORD bytes;
    if(wlen == 0)
    {
      return _base::string_ref("failed to get message from system");
    }
    for(;;)
    {
      auto *p = static_cast<char *>(malloc(allocation)); // NOLINT
      if(p == nullptr)
      {
        return _base::string_ref("failed to get message from system");
      }
      bytes = win32::WideCharToMultiByte(65001 /*CP_UTF8*/, 0, buffer, (int) (wlen + 1), p, (int) allocation, nullptr, nullptr);
      if(bytes != 0)
      {
        char *end = strchr(p, 0);
        while(end[-1] == 10 || end[-1] == 13)
        {
          --end;
        }
        *end = 0; // NOLINT
        return _base::atomic_refcounted_string_ref(p, end - p);
      }
      free(p); // NOLINT
      if(win32::GetLastError() == 0x7a /*ERROR_INSUFFICIENT_BUFFER*/)
      {
        allocation += allocation >> 2;
        continue;
      }
      return _base::string_ref("failed to get message from system");
    }
  }
public:
  //! The value type of the win32 code, which is a `win32::DWORD`
  using value_type = win32::DWORD;
  using _base::string_ref;
public:
  //! Default constructor
  constexpr explicit _win32_code_domain(typename _base::unique_id_type id = 0x8cd18ee72d680f1b) noexcept
      : _base(id)
  {
  }
  _win32_code_domain(const _win32_code_domain &) = default;
  _win32_code_domain(_win32_code_domain &&) = default;
  _win32_code_domain &operator=(const _win32_code_domain &) = default;
  _win32_code_domain &operator=(_win32_code_domain &&) = default;
  ~_win32_code_domain() = default;
  //! Constexpr singleton getter. Returns the constexpr win32_code_domain variable.
  static inline constexpr const _win32_code_domain &get();
  virtual string_ref name() const noexcept override { return string_ref("win32 domain"); } // NOLINT
protected:
  virtual bool _do_failure(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this);
    return static_cast<const win32_code &>(code).value() != 0; // NOLINT
  }
  virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept override // NOLINT
  {
    assert(code1.domain() == *this);
    const auto &c1 = static_cast<const win32_code &>(code1); // NOLINT
    if(code2.domain() == *this)
    {
      const auto &c2 = static_cast<const win32_code &>(code2); // NOLINT
      return c1.value() == c2.value();
    }
    if(code2.domain() == generic_code_domain)
    {
      const auto &c2 = static_cast<const generic_code &>(code2); // NOLINT
      if(static_cast<int>(c2.value()) == _win32_code_to_errno(c1.value()))
      {
        return true;
      }
    }
    return false;
  }
  virtual generic_code _generic_code(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this);
    const auto &c = static_cast<const win32_code &>(code); // NOLINT
    return generic_code(static_cast<errc>(_win32_code_to_errno(c.value())));
  }
  virtual string_ref _do_message(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this);
    const auto &c = static_cast<const win32_code &>(code); // NOLINT
    return _make_string_ref(c.value());
  }
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || 0L
  SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const override // NOLINT
  {
    assert(code.domain() == *this);
    const auto &c = static_cast<const win32_code &>(code); // NOLINT
    throw status_error<_win32_code_domain>(c);
  }
#endif
};
//! (Windows only) A constexpr source variable for the win32 code domain, which is that of `GetLastError()` (Windows). Returned by `_win32_code_domain::get()`.
constexpr _win32_code_domain win32_code_domain;
inline constexpr const _win32_code_domain &_win32_code_domain::get()
{
  return win32_code_domain;
}
namespace mixins
{
  template <class Base> inline win32_code mixin<Base, _win32_code_domain>::current() noexcept { return win32_code(win32::GetLastError()); }
} // namespace mixins
SYSTEM_ERROR2_NAMESPACE_END
#endif
SYSTEM_ERROR2_NAMESPACE_BEGIN
//! \exclude
namespace win32
{
  // A Win32 NTSTATUS
  using NTSTATUS = long;
  // A Win32 HMODULE
  using HMODULE = void *;
  // Used to retrieve where the NTDLL DLL is mapped into memory
  extern HMODULE __stdcall GetModuleHandleW(const wchar_t *lpModuleName);
#pragma comment(lib, "kernel32.lib")
#if (defined(__x86_64__) || defined(_M_X64)) || (defined(__aarch64__) || defined(_M_ARM64))
#pragma comment(linker, "/alternatename:?GetModuleHandleW@win32@system_error2@@YAPEAXPEB_W@Z=GetModuleHandleW")
#elif defined(__x86__) || defined(_M_IX86) || defined(__i386__)
#pragma comment(linker, "/alternatename:?GetModuleHandleW@win32@system_error2@@YGPAXPB_W@Z=__imp__GetModuleHandleW@4")
#elif defined(__arm__) || defined(_M_ARM)
#pragma comment(linker, "/alternatename:?GetModuleHandleW@win32@system_error2@@YAPAXPB_W@Z=GetModuleHandleW")
#else
#error Unknown architecture
#endif
} // namespace win32
class _nt_code_domain;
//! (Windows only) A NT error code, those returned by NT kernel functions.
using nt_code = status_code<_nt_code_domain>;
//! (Windows only) A specialisation of `status_error` for the NT error code domain.
using nt_error = status_error<_nt_code_domain>;
/*! (Windows only) The implementation of the domain for NT error codes, those returned by NT kernel functions.
 */
class _nt_code_domain : public status_code_domain
{
  template <class DomainType> friend class status_code;
  template <class StatusCode> friend class detail::indirecting_domain;
  friend class _com_code_domain;
  using _base = status_code_domain;
  static int _nt_code_to_errno(win32::NTSTATUS c)
  {
    if(c >= 0)
    {
      return 0; // success
    }
    switch(static_cast<unsigned>(c))
    {
case 0x80000002: return EACCES;
case 0x8000000f: return EAGAIN;
case 0x80000010: return EAGAIN;
case 0x80000011: return EBUSY;
case 0xc0000002: return ENOSYS;
case 0xc0000005: return EACCES;
case 0xc0000008: return EINVAL;
case 0xc000000e: return ENOENT;
case 0xc000000f: return ENOENT;
case 0xc0000010: return ENOSYS;
case 0xc0000013: return EAGAIN;
case 0xc0000017: return ENOMEM;
case 0xc000001c: return ENOSYS;
case 0xc000001e: return EACCES;
case 0xc000001f: return EACCES;
case 0xc0000021: return EACCES;
case 0xc0000022: return EACCES;
case 0xc0000024: return EINVAL;
case 0xc0000033: return EINVAL;
case 0xc0000034: return ENOENT;
case 0xc0000035: return EEXIST;
case 0xc0000037: return EINVAL;
case 0xc000003a: return ENOENT;
case 0xc0000040: return ENOMEM;
case 0xc0000041: return EACCES;
case 0xc0000042: return EINVAL;
case 0xc0000043: return EACCES;
case 0xc000004b: return EACCES;
case 0xc0000054: return ENOLCK;
case 0xc0000055: return ENOLCK;
case 0xc0000056: return EACCES;
case 0xc000007f: return ENOSPC;
case 0xc0000087: return ENOMEM;
case 0xc0000097: return ENOMEM;
case 0xc000009b: return ENOENT;
case 0xc000009e: return EAGAIN;
case 0xc00000a2: return EACCES;
case 0xc00000a3: return EAGAIN;
case 0xc00000af: return ENOSYS;
case 0xc00000ba: return EACCES;
case 0xc00000c0: return ENODEV;
case 0xc00000d4: return EXDEV;
case 0xc00000d5: return EACCES;
case 0xc00000fb: return ENOENT;
case 0xc0000101: return ENOTEMPTY;
case 0xc0000103: return EINVAL;
case 0xc0000107: return EBUSY;
case 0xc0000108: return EBUSY;
case 0xc000010a: return EACCES;
case 0xc000011f: return EMFILE;
case 0xc0000120: return ECANCELED;
case 0xc0000121: return EACCES;
case 0xc0000123: return EACCES;
case 0xc0000128: return EINVAL;
case 0xc0000189: return EACCES;
case 0xc00001ad: return ENOMEM;
case 0xc000022d: return EAGAIN;
case 0xc0000235: return EINVAL;
case 0xc000026e: return EAGAIN;
case 0xc000028a: return EACCES;
case 0xc000028b: return EACCES;
case 0xc000028d: return EACCES;
case 0xc000028e: return EACCES;
case 0xc000028f: return EACCES;
case 0xc0000290: return EACCES;
case 0xc000029c: return ENOSYS;
case 0xc00002c5: return EACCES;
case 0xc00002d3: return EAGAIN;
case 0xc00002ea: return EACCES;
case 0xc00002f0: return ENOENT;
case 0xc0000373: return ENOMEM;
case 0xc0000416: return ENOMEM;
case 0xc0000433: return EBUSY;
case 0xc0000434: return EBUSY;
case 0xc0000455: return EINVAL;
case 0xc0000467: return EACCES;
case 0xc0000491: return ENOENT;
case 0xc0000495: return EAGAIN;
case 0xc0000503: return EAGAIN;
case 0xc0000507: return EBUSY;
case 0xc0000512: return EACCES;
case 0xc000070a: return EINVAL;
case 0xc000070b: return EINVAL;
case 0xc000070c: return EINVAL;
case 0xc000070d: return EINVAL;
case 0xc000070e: return EINVAL;
case 0xc000070f: return EINVAL;
case 0xc0000710: return ENOSYS;
case 0xc0000711: return ENOSYS;
case 0xc0000716: return EINVAL;
case 0xc000071b: return ENOSYS;
case 0xc000071d: return ENOSYS;
case 0xc000071e: return ENOSYS;
case 0xc000071f: return ENOSYS;
case 0xc0000720: return ENOSYS;
case 0xc0000721: return ENOSYS;
case 0xc000080f: return EAGAIN;
case 0xc000a203: return EACCES;
    }
    return -1;
  }
  static win32::DWORD _nt_code_to_win32_code(win32::NTSTATUS c) // NOLINT
  {
    if(c >= 0)
    {
      return 0; // success
    }
    switch(static_cast<unsigned>(c))
    {
case 0x80000002: return 0x3e6;
case 0x80000005: return 0xea;
case 0x80000006: return 0x12;
case 0x80000007: return 0x2a3;
case 0x8000000a: return 0x2a4;
case 0x8000000b: return 0x56f;
case 0x8000000c: return 0x2a8;
case 0x8000000d: return 0x12b;
case 0x8000000e: return 0x1c;
case 0x8000000f: return 0x15;
case 0x80000010: return 0x15;
case 0x80000011: return 0xaa;
case 0x80000012: return 0x103;
case 0x80000013: return 0xfe;
case 0x80000014: return 0xff;
case 0x80000015: return 0xff;
case 0x80000016: return 0x456;
case 0x80000017: return 0x2a5;
case 0x80000018: return 0x2a6;
case 0x8000001a: return 0x103;
case 0x8000001b: return 0x44d;
case 0x8000001c: return 0x456;
case 0x8000001d: return 0x457;
case 0x8000001e: return 0x44c;
case 0x8000001f: return 0x44e;
case 0x80000020: return 0x2a7;
case 0x80000021: return 0x44f;
case 0x80000022: return 0x450;
case 0x80000023: return 0x702;
case 0x80000024: return 0x713;
case 0x80000025: return 0x962;
case 0x80000026: return 0x2aa;
case 0x80000027: return 0x10f4;
case 0x80000028: return 0x2ab;
case 0x80000029: return 0x2ac;
case 0x8000002a: return 0x2ad;
case 0x8000002b: return 0x2ae;
case 0x8000002c: return 0x2af;
case 0x8000002d: return 0x2a9;
case 0x8000002e: return 0x321;
case 0x8000002f: return 0x324;
case 0x80000030: return 0xab;
case 0x80000032: return 0xeb;
case 0x80000288: return 0x48d;
case 0x80000289: return 0x48e;
case 0x80000803: return 0x1abb;
case 0x8000a127: return 0x3bdf;
case 0x8000cf00: return 0x16e;
case 0x8000cf04: return 0x16d;
case 0x8000cf05: return 0x176;
case 0x80130001: return 0x13c5;
case 0x80130002: return 0x13c6;
case 0x80130003: return 0x13c7;
case 0x80130004: return 0x13c8;
case 0x80130005: return 0x13c9;
case 0x80190009: return 0x19e5;
case 0x80190029: return 0x1aa0;
case 0x80190031: return 0x1aa2;
case 0x80190041: return 0x1ab3;
case 0x80190042: return 0x1ab4;
case 0x801c0001: return 0x7a;
case 0xc0000001: return 0x1f;
case 0xc0000002: return 0x1;
case 0xc0000003: return 0x57;
case 0xc0000004: return 0x18;
case 0xc0000005: return 0x3e6;
case 0xc0000006: return 0x3e7;
case 0xc0000007: return 0x5ae;
case 0xc0000008: return 0x6;
case 0xc0000009: return 0x3e9;
case 0xc000000a: return 0xc1;
case 0xc000000b: return 0x57;
case 0xc000000c: return 0x21d;
case 0xc000000d: return 0x57;
case 0xc000000e: return 0x2;
case 0xc000000f: return 0x2;
case 0xc0000010: return 0x1;
case 0xc0000011: return 0x26;
case 0xc0000012: return 0x22;
case 0xc0000013: return 0x15;
case 0xc0000014: return 0x6f9;
case 0xc0000015: return 0x1b;
case 0xc0000016: return 0xea;
case 0xc0000017: return 0x8;
case 0xc0000018: return 0x1e7;
case 0xc0000019: return 0x1e7;
case 0xc000001a: return 0x57;
case 0xc000001b: return 0x57;
case 0xc000001c: return 0x1;
case 0xc000001e: return 0x5;
case 0xc000001f: return 0x5;
case 0xc0000020: return 0xc1;
case 0xc0000021: return 0x5;
case 0xc0000022: return 0x5;
case 0xc0000023: return 0x7a;
case 0xc0000024: return 0x6;
case 0xc0000027: return 0x21e;
case 0xc0000028: return 0x21f;
case 0xc0000029: return 0x220;
case 0xc000002a: return 0x9e;
case 0xc000002c: return 0x1e7;
case 0xc000002d: return 0x1e7;
case 0xc000002e: return 0x221;
case 0xc000002f: return 0x222;
case 0xc0000030: return 0x57;
case 0xc0000031: return 0x223;
case 0xc0000032: return 0x571;
case 0xc0000033: return 0x7b;
case 0xc0000034: return 0x2;
case 0xc0000035: return 0xb7;
case 0xc0000036: return 0x72a;
case 0xc0000037: return 0x6;
case 0xc0000038: return 0x224;
case 0xc0000039: return 0xa1;
case 0xc000003a: return 0x3;
case 0xc000003b: return 0xa1;
case 0xc000003c: return 0x45d;
case 0xc000003d: return 0x45d;
case 0xc000003e: return 0x17;
case 0xc000003f: return 0x17;
case 0xc0000040: return 0x8;
case 0xc0000041: return 0x5;
case 0xc0000042: return 0x6;
case 0xc0000043: return 0x20;
case 0xc0000044: return 0x718;
case 0xc0000045: return 0x57;
case 0xc0000046: return 0x120;
case 0xc0000047: return 0x12a;
case 0xc0000048: return 0x57;
case 0xc0000049: return 0x57;
case 0xc000004a: return 0x9c;
case 0xc000004b: return 0x5;
case 0xc000004c: return 0x57;
case 0xc000004d: return 0x57;
case 0xc000004e: return 0x57;
case 0xc000004f: return 0x11a;
case 0xc0000050: return 0xff;
case 0xc0000051: return 0x570;
case 0xc0000052: return 0x570;
case 0xc0000053: return 0x570;
case 0xc0000054: return 0x21;
case 0xc0000055: return 0x21;
case 0xc0000056: return 0x5;
case 0xc0000057: return 0x32;
case 0xc0000058: return 0x519;
case 0xc0000059: return 0x51a;
case 0xc000005a: return 0x51b;
case 0xc000005b: return 0x51c;
case 0xc000005c: return 0x51d;
case 0xc000005d: return 0x51e;
case 0xc000005e: return 0x51f;
case 0xc000005f: return 0x520;
case 0xc0000060: return 0x521;
case 0xc0000061: return 0x522;
case 0xc0000062: return 0x523;
case 0xc0000063: return 0x524;
case 0xc0000064: return 0x525;
case 0xc0000065: return 0x526;
case 0xc0000066: return 0x527;
case 0xc0000067: return 0x528;
case 0xc0000068: return 0x529;
case 0xc0000069: return 0x52a;
case 0xc000006a: return 0x56;
case 0xc000006b: return 0x52c;
case 0xc000006c: return 0x52d;
case 0xc000006d: return 0x52e;
case 0xc000006e: return 0x52f;
case 0xc000006f: return 0x530;
case 0xc0000070: return 0x531;
case 0xc0000071: return 0x532;
case 0xc0000072: return 0x533;
case 0xc0000073: return 0x534;
case 0xc0000074: return 0x535;
case 0xc0000075: return 0x536;
case 0xc0000076: return 0x537;
case 0xc0000077: return 0x538;
case 0xc0000078: return 0x539;
case 0xc0000079: return 0x53a;
case 0xc000007a: return 0x7f;
case 0xc000007b: return 0xc1;
case 0xc000007c: return 0x3f0;
case 0xc000007d: return 0x53c;
case 0xc000007e: return 0x9e;
case 0xc000007f: return 0x70;
case 0xc0000080: return 0x53d;
case 0xc0000081: return 0x53e;
case 0xc0000082: return 0x44;
case 0xc0000083: return 0x103;
case 0xc0000084: return 0x53f;
case 0xc0000085: return 0x103;
case 0xc0000086: return 0x9a;
case 0xc0000087: return 0xe;
case 0xc0000088: return 0x1e7;
case 0xc0000089: return 0x714;
case 0xc000008a: return 0x715;
case 0xc000008b: return 0x716;
case 0xc0000095: return 0x216;
case 0xc0000097: return 0x8;
case 0xc0000098: return 0x3ee;
case 0xc0000099: return 0x540;
case 0xc000009a: return 0x5aa;
case 0xc000009b: return 0x3;
case 0xc000009c: return 0x17;
case 0xc000009d: return 0x48f;
case 0xc000009e: return 0x15;
case 0xc000009f: return 0x1e7;
case 0xc00000a0: return 0x1e7;
case 0xc00000a1: return 0x5ad;
case 0xc00000a2: return 0x13;
case 0xc00000a3: return 0x15;
case 0xc00000a4: return 0x541;
case 0xc00000a5: return 0x542;
case 0xc00000a6: return 0x543;
case 0xc00000a7: return 0x544;
case 0xc00000a8: return 0x545;
case 0xc00000a9: return 0x57;
case 0xc00000aa: return 0x225;
case 0xc00000ab: return 0xe7;
case 0xc00000ac: return 0xe7;
case 0xc00000ad: return 0xe6;
case 0xc00000ae: return 0xe7;
case 0xc00000af: return 0x1;
case 0xc00000b0: return 0xe9;
case 0xc00000b1: return 0xe8;
case 0xc00000b2: return 0x217;
case 0xc00000b3: return 0x218;
case 0xc00000b4: return 0xe6;
case 0xc00000b5: return 0x79;
case 0xc00000b6: return 0x26;
case 0xc00000b7: return 0x226;
case 0xc00000b8: return 0x227;
case 0xc00000b9: return 0x228;
case 0xc00000ba: return 0x5;
case 0xc00000bb: return 0x32;
case 0xc00000bc: return 0x33;
case 0xc00000bd: return 0x34;
case 0xc00000be: return 0x35;
case 0xc00000bf: return 0x36;
case 0xc00000c0: return 0x37;
case 0xc00000c1: return 0x38;
case 0xc00000c2: return 0x39;
case 0xc00000c3: return 0x3a;
case 0xc00000c4: return 0x3b;
case 0xc00000c5: return 0x3c;
case 0xc00000c6: return 0x3d;
case 0xc00000c7: return 0x3e;
case 0xc00000c8: return 0x3f;
case 0xc00000c9: return 0x40;
case 0xc00000ca: return 0x41;
case 0xc00000cb: return 0x42;
case 0xc00000cc: return 0x43;
case 0xc00000cd: return 0x44;
case 0xc00000ce: return 0x45;
case 0xc00000cf: return 0x46;
case 0xc00000d0: return 0x47;
case 0xc00000d1: return 0x48;
case 0xc00000d2: return 0x58;
case 0xc00000d3: return 0x229;
case 0xc00000d4: return 0x11;
case 0xc00000d5: return 0x5;
case 0xc00000d6: return 0xf0;
case 0xc00000d7: return 0x546;
case 0xc00000d8: return 0x22a;
case 0xc00000d9: return 0xe8;
case 0xc00000da: return 0x547;
case 0xc00000db: return 0x22b;
case 0xc00000dc: return 0x548;
case 0xc00000dd: return 0x549;
case 0xc00000de: return 0x54a;
case 0xc00000df: return 0x54b;
case 0xc00000e0: return 0x54c;
case 0xc00000e1: return 0x54d;
case 0xc00000e2: return 0x12c;
case 0xc00000e3: return 0x12d;
case 0xc00000e4: return 0x54e;
case 0xc00000e5: return 0x54f;
case 0xc00000e6: return 0x550;
case 0xc00000e7: return 0x551;
case 0xc00000e8: return 0x6f8;
case 0xc00000e9: return 0x45d;
case 0xc00000ea: return 0x22c;
case 0xc00000eb: return 0x22d;
case 0xc00000ec: return 0x22e;
case 0xc00000ed: return 0x552;
case 0xc00000ee: return 0x553;
case 0xc00000ef: return 0x57;
case 0xc00000f0: return 0x57;
case 0xc00000f1: return 0x57;
case 0xc00000f2: return 0x57;
case 0xc00000f3: return 0x57;
case 0xc00000f4: return 0x57;
case 0xc00000f5: return 0x57;
case 0xc00000f6: return 0x57;
case 0xc00000f7: return 0x57;
case 0xc00000f8: return 0x57;
case 0xc00000f9: return 0x57;
case 0xc00000fa: return 0x57;
case 0xc00000fb: return 0x3;
case 0xc00000fc: return 0x420;
case 0xc00000fd: return 0x3e9;
case 0xc00000fe: return 0x554;
case 0xc00000ff: return 0x22f;
case 0xc0000100: return 0xcb;
case 0xc0000101: return 0x91;
case 0xc0000102: return 0x570;
case 0xc0000103: return 0x10b;
case 0xc0000104: return 0x555;
case 0xc0000105: return 0x556;
case 0xc0000106: return 0xce;
case 0xc0000107: return 0x961;
case 0xc0000108: return 0x964;
case 0xc000010a: return 0x5;
case 0xc000010b: return 0x557;
case 0xc000010c: return 0x230;
case 0xc000010d: return 0x558;
case 0xc000010e: return 0x420;
case 0xc000010f: return 0x21a;
case 0xc0000110: return 0x21a;
case 0xc0000111: return 0x21a;
case 0xc0000112: return 0x21a;
case 0xc0000113: return 0x21a;
case 0xc0000114: return 0x21a;
case 0xc0000115: return 0x21a;
case 0xc0000116: return 0x21a;
case 0xc0000117: return 0x5a4;
case 0xc0000118: return 0x231;
case 0xc0000119: return 0x233;
case 0xc000011a: return 0x234;
case 0xc000011b: return 0xc1;
case 0xc000011c: return 0x559;
case 0xc000011d: return 0x55a;
case 0xc000011e: return 0x3ee;
case 0xc000011f: return 0x4;
case 0xc0000120: return 0x3e3;
case 0xc0000121: return 0x5;
case 0xc0000122: return 0x4ba;
case 0xc0000123: return 0x5;
case 0xc0000124: return 0x55b;
case 0xc0000125: return 0x55c;
case 0xc0000126: return 0x55d;
case 0xc0000127: return 0x55e;
case 0xc0000128: return 0x6;
case 0xc0000129: return 0x235;
case 0xc000012a: return 0x236;
case 0xc000012b: return 0x55f;
case 0xc000012c: return 0x237;
case 0xc000012d: return 0x5af;
case 0xc000012e: return 0xc1;
case 0xc000012f: return 0xc1;
case 0xc0000130: return 0xc1;
case 0xc0000131: return 0xc1;
case 0xc0000132: return 0x238;
case 0xc0000133: return 0x576;
case 0xc0000134: return 0x239;
case 0xc0000135: return 0x7e;
case 0xc0000136: return 0x23a;
case 0xc0000137: return 0x23b;
case 0xc0000138: return 0xb6;
case 0xc0000139: return 0x7f;
case 0xc000013a: return 0x23c;
case 0xc000013b: return 0x40;
case 0xc000013c: return 0x40;
case 0xc000013d: return 0x33;
case 0xc000013e: return 0x3b;
case 0xc000013f: return 0x3b;
case 0xc0000140: return 0x3b;
case 0xc0000141: return 0x3b;
case 0xc0000142: return 0x45a;
case 0xc0000143: return 0x23d;
case 0xc0000144: return 0x23e;
case 0xc0000145: return 0x23f;
case 0xc0000146: return 0x240;
case 0xc0000147: return 0x242;
case 0xc0000148: return 0x7c;
case 0xc0000149: return 0x56;
case 0xc000014a: return 0x243;
case 0xc000014b: return 0x6d;
case 0xc000014c: return 0x3f1;
case 0xc000014d: return 0x3f8;
case 0xc000014e: return 0x244;
case 0xc000014f: return 0x3ed;
case 0xc0000150: return 0x45e;
case 0xc0000151: return 0x560;
case 0xc0000152: return 0x561;
case 0xc0000153: return 0x562;
case 0xc0000154: return 0x563;
case 0xc0000155: return 0x564;
case 0xc0000156: return 0x565;
case 0xc0000157: return 0x566;
case 0xc0000158: return 0x567;
case 0xc0000159: return 0x3ef;
case 0xc000015a: return 0x568;
case 0xc000015b: return 0x569;
case 0xc000015c: return 0x3f9;
case 0xc000015d: return 0x56a;
case 0xc000015e: return 0x245;
case 0xc000015f: return 0x45d;
case 0xc0000160: return 0x4db;
case 0xc0000161: return 0x246;
case 0xc0000162: return 0x459;
case 0xc0000163: return 0x247;
case 0xc0000164: return 0x248;
case 0xc0000165: return 0x462;
case 0xc0000166: return 0x463;
case 0xc0000167: return 0x464;
case 0xc0000168: return 0x465;
case 0xc0000169: return 0x466;
case 0xc000016a: return 0x467;
case 0xc000016b: return 0x468;
case 0xc000016c: return 0x45f;
case 0xc000016d: return 0x45d;
case 0xc000016e: return 0x249;
case 0xc0000172: return 0x451;
case 0xc0000173: return 0x452;
case 0xc0000174: return 0x453;
case 0xc0000175: return 0x454;
case 0xc0000176: return 0x455;
case 0xc0000177: return 0x469;
case 0xc0000178: return 0x458;
case 0xc000017a: return 0x56b;
case 0xc000017b: return 0x56c;
case 0xc000017c: return 0x3fa;
case 0xc000017d: return 0x3fb;
case 0xc000017e: return 0x56d;
case 0xc000017f: return 0x56e;
case 0xc0000180: return 0x3fc;
case 0xc0000181: return 0x3fd;
case 0xc0000182: return 0x57;
case 0xc0000183: return 0x45d;
case 0xc0000184: return 0x16;
case 0xc0000185: return 0x45d;
case 0xc0000186: return 0x45d;
case 0xc0000187: return 0x24a;
case 0xc0000188: return 0x5de;
case 0xc0000189: return 0x13;
case 0xc000018a: return 0x6fa;
case 0xc000018b: return 0x6fb;
case 0xc000018c: return 0x6fc;
case 0xc000018d: return 0x6fd;
case 0xc000018e: return 0x5dc;
case 0xc000018f: return 0x5dd;
case 0xc0000190: return 0x6fe;
case 0xc0000191: return 0x24b;
case 0xc0000192: return 0x700;
case 0xc0000193: return 0x701;
case 0xc0000194: return 0x46b;
case 0xc0000195: return 0x4c3;
case 0xc0000196: return 0x4c4;
case 0xc0000197: return 0x5df;
case 0xc0000198: return 0x70f;
case 0xc0000199: return 0x710;
case 0xc000019a: return 0x711;
case 0xc000019b: return 0x712;
case 0xc000019c: return 0x24c;
case 0xc000019d: return 0x420;
case 0xc000019e: return 0x130;
case 0xc000019f: return 0x131;
case 0xc00001a0: return 0x132;
case 0xc00001a1: return 0x133;
case 0xc00001a2: return 0x325;
case 0xc00001a3: return 0x134;
case 0xc00001a4: return 0x135;
case 0xc00001a5: return 0x136;
case 0xc00001a6: return 0x137;
case 0xc00001a7: return 0x139;
case 0xc00001a8: return 0x1abb;
case 0xc00001a9: return 0x32;
case 0xc00001aa: return 0x3d54;
case 0xc00001ab: return 0x329;
case 0xc00001ac: return 0x678;
case 0xc00001ad: return 0x8;
case 0xc00001ae: return 0x2f7;
case 0xc00001af: return 0x32d;
case 0xc0000201: return 0x41;
case 0xc0000202: return 0x572;
case 0xc0000203: return 0x3b;
case 0xc0000204: return 0x717;
case 0xc0000205: return 0x46a;
case 0xc0000206: return 0x6f8;
case 0xc0000207: return 0x4be;
case 0xc0000208: return 0x4be;
case 0xc0000209: return 0x44;
case 0xc000020a: return 0x34;
case 0xc000020b: return 0x40;
case 0xc000020c: return 0x40;
case 0xc000020d: return 0x40;
case 0xc000020e: return 0x44;
case 0xc000020f: return 0x3b;
case 0xc0000210: return 0x3b;
case 0xc0000211: return 0x3b;
case 0xc0000212: return 0x3b;
case 0xc0000213: return 0x3b;
case 0xc0000214: return 0x3b;
case 0xc0000215: return 0x3b;
case 0xc0000216: return 0x32;
case 0xc0000217: return 0x32;
case 0xc0000218: return 0x24d;
case 0xc0000219: return 0x24e;
case 0xc000021a: return 0x24f;
case 0xc000021b: return 0x250;
case 0xc000021c: return 0x17e6;
case 0xc000021d: return 0x251;
case 0xc000021e: return 0x252;
case 0xc000021f: return 0x253;
case 0xc0000220: return 0x46c;
case 0xc0000221: return 0xc1;
case 0xc0000222: return 0x254;
case 0xc0000223: return 0x255;
case 0xc0000224: return 0x773;
case 0xc0000225: return 0x490;
case 0xc0000226: return 0x256;
case 0xc0000227: return 0x4ff;
case 0xc0000228: return 0x257;
case 0xc0000229: return 0x57;
case 0xc000022a: return 0x1392;
case 0xc000022b: return 0x1392;
case 0xc000022c: return 0x258;
case 0xc000022d: return 0x4d5;
case 0xc000022e: return 0x259;
case 0xc000022f: return 0x25a;
case 0xc0000230: return 0x492;
case 0xc0000231: return 0x25b;
case 0xc0000232: return 0x25c;
case 0xc0000233: return 0x774;
case 0xc0000234: return 0x775;
case 0xc0000235: return 0x6;
case 0xc0000236: return 0x4c9;
case 0xc0000237: return 0x4ca;
case 0xc0000238: return 0x4cb;
case 0xc0000239: return 0x4cc;
case 0xc000023a: return 0x4cd;
case 0xc000023b: return 0x4ce;
case 0xc000023c: return 0x4cf;
case 0xc000023d: return 0x4d0;
case 0xc000023e: return 0x4d1;
case 0xc000023f: return 0x4d2;
case 0xc0000240: return 0x4d3;
case 0xc0000241: return 0x4d4;
case 0xc0000242: return 0x25d;
case 0xc0000243: return 0x4c8;
case 0xc0000244: return 0x25e;
case 0xc0000245: return 0x25f;
case 0xc0000246: return 0x4d6;
case 0xc0000247: return 0x4d7;
case 0xc0000248: return 0x4d8;
case 0xc0000249: return 0xc1;
case 0xc0000250: return 0x260;
case 0xc0000251: return 0x261;
case 0xc0000252: return 0x262;
case 0xc0000253: return 0x4d4;
case 0xc0000254: return 0x263;
case 0xc0000255: return 0x264;
case 0xc0000256: return 0x265;
case 0xc0000257: return 0x4d0;
case 0xc0000258: return 0x266;
case 0xc0000259: return 0x573;
case 0xc000025a: return 0x267;
case 0xc000025b: return 0x268;
case 0xc000025c: return 0x269;
case 0xc000025e: return 0x422;
case 0xc000025f: return 0x26a;
case 0xc0000260: return 0x26b;
case 0xc0000261: return 0x26c;
case 0xc0000262: return 0xb6;
case 0xc0000263: return 0x7f;
case 0xc0000264: return 0x120;
case 0xc0000265: return 0x476;
case 0xc0000266: return 0x26d;
case 0xc0000267: return 0x10fe;
case 0xc0000268: return 0x26e;
case 0xc0000269: return 0x26f;
case 0xc000026a: return 0x1b8e;
case 0xc000026b: return 0x270;
case 0xc000026c: return 0x7d1;
case 0xc000026d: return 0x4b1;
case 0xc000026e: return 0x15;
case 0xc000026f: return 0x21c;
case 0xc0000270: return 0x21c;
case 0xc0000271: return 0x271;
case 0xc0000272: return 0x491;
case 0xc0000273: return 0x272;
case 0xc0000275: return 0x1126;
case 0xc0000276: return 0x1129;
case 0xc0000277: return 0x112a;
case 0xc0000278: return 0x1128;
case 0xc0000279: return 0x780;
case 0xc000027a: return 0x291;
case 0xc000027b: return 0x54f;
case 0xc000027c: return 0x54f;
case 0xc0000280: return 0x781;
case 0xc0000281: return 0xa1;
case 0xc0000282: return 0x273;
case 0xc0000283: return 0x488;
case 0xc0000284: return 0x489;
case 0xc0000285: return 0x48a;
case 0xc0000286: return 0x48b;
case 0xc0000287: return 0x48c;
case 0xc000028a: return 0x5;
case 0xc000028b: return 0x5;
case 0xc000028c: return 0x284;
case 0xc000028d: return 0x5;
case 0xc000028e: return 0x5;
case 0xc000028f: return 0x5;
case 0xc0000290: return 0x5;
case 0xc0000291: return 0x1777;
case 0xc0000292: return 0x1778;
case 0xc0000293: return 0x1772;
case 0xc0000295: return 0x1068;
case 0xc0000296: return 0x1069;
case 0xc0000297: return 0x106a;
case 0xc0000298: return 0x106b;
case 0xc0000299: return 0x201a;
case 0xc000029a: return 0x201b;
case 0xc000029b: return 0x201c;
case 0xc000029c: return 0x1;
case 0xc000029d: return 0x10ff;
case 0xc000029e: return 0x1100;
case 0xc000029f: return 0x494;
case 0xc00002a0: return 0x274;
case 0xc00002a1: return 0x200a;
case 0xc00002a2: return 0x200b;
case 0xc00002a3: return 0x200c;
case 0xc00002a4: return 0x200d;
case 0xc00002a5: return 0x200e;
case 0xc00002a6: return 0x200f;
case 0xc00002a7: return 0x2010;
case 0xc00002a8: return 0x2011;
case 0xc00002a9: return 0x2012;
case 0xc00002aa: return 0x2013;
case 0xc00002ab: return 0x2014;
case 0xc00002ac: return 0x2015;
case 0xc00002ad: return 0x2016;
case 0xc00002ae: return 0x2017;
case 0xc00002af: return 0x2018;
case 0xc00002b0: return 0x2019;
case 0xc00002b1: return 0x211e;
case 0xc00002b2: return 0x1127;
case 0xc00002b3: return 0x275;
case 0xc00002b4: return 0x276;
case 0xc00002b5: return 0x277;
case 0xc00002b6: return 0x651;
case 0xc00002b7: return 0x49a;
case 0xc00002b8: return 0x49b;
case 0xc00002b9: return 0x278;
case 0xc00002ba: return 0x2047;
case 0xc00002c1: return 0x2024;
case 0xc00002c2: return 0x279;
case 0xc00002c3: return 0x575;
case 0xc00002c4: return 0x27a;
case 0xc00002c5: return 0x3e6;
case 0xc00002c6: return 0x1075;
case 0xc00002c7: return 0x1076;
case 0xc00002c8: return 0x27b;
case 0xc00002c9: return 0x4ed;
case 0xc00002ca: return 0x10e8;
case 0xc00002cb: return 0x2138;
case 0xc00002cc: return 0x4e3;
case 0xc00002cd: return 0x2139;
case 0xc00002ce: return 0x27c;
case 0xc00002cf: return 0x49d;
case 0xc00002d0: return 0x213a;
case 0xc00002d1: return 0x27d;
case 0xc00002d2: return 0x27e;
case 0xc00002d3: return 0x15;
case 0xc00002d4: return 0x2141;
case 0xc00002d5: return 0x2142;
case 0xc00002d6: return 0x2143;
case 0xc00002d7: return 0x2144;
case 0xc00002d8: return 0x2145;
case 0xc00002d9: return 0x2146;
case 0xc00002da: return 0x2147;
case 0xc00002db: return 0x2148;
case 0xc00002dc: return 0x2149;
case 0xc00002dd: return 0x32;
case 0xc00002de: return 0x27f;
case 0xc00002df: return 0x2151;
case 0xc00002e0: return 0x2152;
case 0xc00002e1: return 0x2153;
case 0xc00002e2: return 0x2154;
case 0xc00002e3: return 0x215d;
case 0xc00002e4: return 0x2163;
case 0xc00002e5: return 0x2164;
case 0xc00002e6: return 0x2165;
case 0xc00002e7: return 0x216d;
case 0xc00002e8: return 0x280;
case 0xc00002e9: return 0x577;
case 0xc00002ea: return 0x52;
case 0xc00002eb: return 0x281;
case 0xc00002ec: return 0x2171;
case 0xc00002ed: return 0x2172;
case 0xc00002f0: return 0x2;
case 0xc00002fe: return 0x45b;
case 0xc00002ff: return 0x4e7;
case 0xc0000300: return 0x4e6;
case 0xc0000301: return 0x106f;
case 0xc0000302: return 0x1074;
case 0xc0000303: return 0x106e;
case 0xc0000304: return 0x12e;
case 0xc000030c: return 0x792;
case 0xc000030d: return 0x793;
case 0xc0000320: return 0x4ef;
case 0xc0000321: return 0x4f0;
case 0xc0000350: return 0x4e8;
case 0xc0000352: return 0x177d;
case 0xc0000353: return 0x282;
case 0xc0000354: return 0x504;
case 0xc0000355: return 0x283;
case 0xc0000357: return 0x217c;
case 0xc0000358: return 0x2182;
case 0xc0000359: return 0xc1;
case 0xc000035a: return 0xc1;
case 0xc000035c: return 0x572;
case 0xc000035d: return 0x4eb;
case 0xc000035f: return 0x286;
case 0xc0000361: return 0x4ec;
case 0xc0000362: return 0x4ec;
case 0xc0000363: return 0x4ec;
case 0xc0000364: return 0x4ec;
case 0xc0000365: return 0x287;
case 0xc0000366: return 0x288;
case 0xc0000368: return 0x289;
case 0xc0000369: return 0x28a;
case 0xc000036a: return 0x28b;
case 0xc000036b: return 0x4fb;
case 0xc000036c: return 0x4fb;
case 0xc000036d: return 0x28c;
case 0xc000036e: return 0x28d;
case 0xc000036f: return 0x4fc;
case 0xc0000371: return 0x21ac;
case 0xc0000372: return 0x312;
case 0xc0000373: return 0x8;
case 0xc0000374: return 0x54f;
case 0xc0000388: return 0x4f1;
case 0xc000038e: return 0x28e;
case 0xc0000401: return 0x78c;
case 0xc0000402: return 0x78d;
case 0xc0000403: return 0x78e;
case 0xc0000404: return 0x217b;
case 0xc0000405: return 0x219d;
case 0xc0000406: return 0x219f;
case 0xc0000407: return 0x28f;
case 0xc0000408: return 0x52e;
case 0xc0000409: return 0x502;
case 0xc0000410: return 0x503;
case 0xc0000411: return 0x290;
case 0xc0000412: return 0x505;
case 0xc0000413: return 0x78f;
case 0xc0000414: return 0x506;
case 0xc0000416: return 0x8;
case 0xc0000417: return 0x508;
case 0xc0000418: return 0x791;
case 0xc0000419: return 0x215b;
case 0xc000041a: return 0x21ba;
case 0xc000041b: return 0x21bb;
case 0xc000041c: return 0x21bc;
case 0xc000041d: return 0x2c9;
case 0xc0000420: return 0x29c;
case 0xc0000421: return 0x219;
case 0xc0000423: return 0x300;
case 0xc0000424: return 0x4fb;
case 0xc0000425: return 0x3fa;
case 0xc0000426: return 0x301;
case 0xc0000427: return 0x299;
case 0xc0000428: return 0x241;
case 0xc0000429: return 0x307;
case 0xc000042a: return 0x308;
case 0xc000042b: return 0x50c;
case 0xc000042c: return 0x2e4;
case 0xc0000432: return 0x509;
case 0xc0000433: return 0xaa;
case 0xc0000434: return 0xaa;
case 0xc0000435: return 0x4c8;
case 0xc0000441: return 0x1781;
case 0xc0000442: return 0x1782;
case 0xc0000443: return 0x1783;
case 0xc0000444: return 0x1784;
case 0xc0000445: return 0x1785;
case 0xc0000446: return 0x513;
case 0xc0000450: return 0x50b;
case 0xc0000451: return 0x3b92;
case 0xc0000452: return 0x3bc3;
case 0xc0000453: return 0x5bb;
case 0xc0000454: return 0x5be;
case 0xc0000455: return 0x6;
case 0xc0000456: return 0x57;
case 0xc0000457: return 0x57;
case 0xc0000458: return 0x57;
case 0xc0000459: return 0xbea;
case 0xc0000460: return 0x138;
case 0xc0000461: return 0x13a;
case 0xc0000462: return 0x3cfc;
case 0xc0000463: return 0x13c;
case 0xc0000464: return 0x141;
case 0xc0000465: return 0x13b;
case 0xc0000466: return 0x40;
case 0xc0000467: return 0x20;
case 0xc0000468: return 0x142;
case 0xc0000469: return 0x3d00;
case 0xc000046a: return 0x151;
case 0xc000046b: return 0x152;
case 0xc000046c: return 0x153;
case 0xc000046d: return 0x156;
case 0xc000046e: return 0x157;
case 0xc000046f: return 0x158;
case 0xc0000470: return 0x143;
case 0xc0000471: return 0x144;
case 0xc0000472: return 0x146;
case 0xc0000473: return 0x14b;
case 0xc0000474: return 0x147;
case 0xc0000475: return 0x148;
case 0xc0000476: return 0x149;
case 0xc0000477: return 0x14a;
case 0xc0000478: return 0x14c;
case 0xc0000479: return 0x14d;
case 0xc000047a: return 0x14e;
case 0xc000047b: return 0x14f;
case 0xc000047c: return 0x150;
case 0xc000047d: return 0x5b4;
case 0xc000047e: return 0x3d07;
case 0xc000047f: return 0x3d08;
case 0xc0000480: return 0x40;
case 0xc0000481: return 0x7e;
case 0xc0000482: return 0x7e;
case 0xc0000483: return 0x1e3;
case 0xc0000486: return 0x159;
case 0xc0000487: return 0x1f;
case 0xc0000488: return 0x15a;
case 0xc0000489: return 0x3d0f;
case 0xc000048a: return 0x32a;
case 0xc000048b: return 0x32c;
case 0xc000048c: return 0x15b;
case 0xc000048d: return 0x15c;
case 0xc000048e: return 0x162;
case 0xc000048f: return 0x15d;
case 0xc0000490: return 0x491;
case 0xc0000491: return 0x2;
case 0xc0000492: return 0x490;
case 0xc0000493: return 0x492;
case 0xc0000494: return 0x307;
case 0xc0000495: return 0x15;
case 0xc0000496: return 0x163;
case 0xc0000497: return 0x3d5a;
case 0xc0000499: return 0x167;
case 0xc000049a: return 0x168;
case 0xc000049b: return 0x12e;
case 0xc000049c: return 0x169;
case 0xc000049d: return 0x16f;
case 0xc000049e: return 0x170;
case 0xc000049f: return 0x49f;
case 0xc00004a0: return 0x4a0;
case 0xc00004a1: return 0x18f;
case 0xc0000500: return 0x60e;
case 0xc0000501: return 0x60f;
case 0xc0000502: return 0x610;
case 0xc0000503: return 0x15;
case 0xc0000504: return 0x13f;
case 0xc0000505: return 0x140;
case 0xc0000506: return 0x5bf;
case 0xc0000507: return 0xaa;
case 0xc0000508: return 0x5e0;
case 0xc0000509: return 0x5e1;
case 0xc000050b: return 0x112b;
case 0xc000050e: return 0x115c;
case 0xc000050f: return 0x10d3;
case 0xc0000510: return 0x4df;
case 0xc0000511: return 0x32e;
case 0xc0000512: return 0x5;
case 0xc0000513: return 0x180;
case 0xc0000514: return 0x115d;
case 0xc0000602: return 0x675;
case 0xc0000604: return 0x677;
case 0xc0000606: return 0x679;
case 0xc000060a: return 0x67c;
case 0xc000060b: return 0x67d;
case 0xc0000700: return 0x54f;
case 0xc0000701: return 0x54f;
case 0xc0000702: return 0x57;
case 0xc0000703: return 0x54f;
case 0xc0000704: return 0x32;
case 0xc0000705: return 0x57;
case 0xc0000706: return 0x57;
case 0xc0000707: return 0x32;
case 0xc0000708: return 0x54f;
case 0xc0000709: return 0x30b;
case 0xc000070a: return 0x6;
case 0xc000070b: return 0x6;
case 0xc000070c: return 0x6;
case 0xc000070d: return 0x6;
case 0xc000070e: return 0x6;
case 0xc000070f: return 0x6;
case 0xc0000710: return 0x1;
case 0xc0000711: return 0x1;
case 0xc0000712: return 0x50d;
case 0xc0000713: return 0x310;
case 0xc0000714: return 0x52e;
case 0xc0000715: return 0x5b7;
case 0xc0000716: return 0x7b;
case 0xc0000717: return 0x459;
case 0xc0000718: return 0x54f;
case 0xc0000719: return 0x54f;
case 0xc000071a: return 0x54f;
case 0xc000071b: return 0x1;
case 0xc000071c: return 0x57;
case 0xc000071d: return 0x1;
case 0xc000071e: return 0x1;
case 0xc000071f: return 0x1;
case 0xc0000720: return 0x1;
case 0xc0000721: return 0x1;
case 0xc0000722: return 0x72b;
case 0xc0000723: return 0x1f;
case 0xc0000724: return 0x1f;
case 0xc0000725: return 0x1f;
case 0xc0000726: return 0x1f;
case 0xc0000800: return 0x30c;
case 0xc0000801: return 0x21a4;
case 0xc0000802: return 0x50f;
case 0xc0000804: return 0x510;
case 0xc0000805: return 0x1ac1;
case 0xc0000806: return 0x1ac3;
case 0xc0000808: return 0x319;
case 0xc0000809: return 0x31a;
case 0xc000080a: return 0x31b;
case 0xc000080b: return 0x31c;
case 0xc000080c: return 0x31d;
case 0xc000080d: return 0x31e;
case 0xc000080e: return 0x31f;
case 0xc000080f: return 0x4d5;
case 0xc0000810: return 0x328;
case 0xc0000811: return 0x54f;
case 0xc0000901: return 0xdc;
case 0xc0000902: return 0xdd;
case 0xc0000903: return 0xde;
case 0xc0000904: return 0xdf;
case 0xc0000905: return 0xe0;
case 0xc0000906: return 0xe1;
case 0xc0000907: return 0xe2;
case 0xc0000908: return 0x317;
case 0xc0000909: return 0x322;
case 0xc0000910: return 0x326;
case 0xc0009898: return 0x29e;
case 0xc000a002: return 0x17;
case 0xc000a003: return 0x139f;
case 0xc000a004: return 0x154;
case 0xc000a005: return 0x155;
case 0xc000a006: return 0x32b;
case 0xc000a007: return 0x32;
case 0xc000a010: return 0xea;
case 0xc000a011: return 0xea;
case 0xc000a012: return 0x4d0;
case 0xc000a013: return 0x32;
case 0xc000a014: return 0x4d1;
case 0xc000a080: return 0x314;
case 0xc000a081: return 0x315;
case 0xc000a082: return 0x316;
case 0xc000a083: return 0x5b9;
case 0xc000a084: return 0x5ba;
case 0xc000a085: return 0x5bc;
case 0xc000a086: return 0x5bd;
case 0xc000a087: return 0x21bd;
case 0xc000a088: return 0x21be;
case 0xc000a089: return 0x21c6;
case 0xc000a100: return 0x3bc4;
case 0xc000a101: return 0x3bc5;
case 0xc000a121: return 0x3bd9;
case 0xc000a122: return 0x3bda;
case 0xc000a123: return 0x3bdb;
case 0xc000a124: return 0x3bdc;
case 0xc000a125: return 0x3bdd;
case 0xc000a126: return 0x3bde;
case 0xc000a141: return 0x3c28;
case 0xc000a142: return 0x3c29;
case 0xc000a143: return 0x3c2a;
case 0xc000a145: return 0x3c2b;
case 0xc000a146: return 0x3c2c;
case 0xc000a200: return 0x109a;
case 0xc000a201: return 0x109c;
case 0xc000a202: return 0x109d;
case 0xc000a203: return 0x5;
case 0xc000a281: return 0x1130;
case 0xc000a282: return 0x1131;
case 0xc000a283: return 0x1132;
case 0xc000a284: return 0x1133;
case 0xc000a285: return 0x1134;
case 0xc000a2a1: return 0x1158;
case 0xc000a2a2: return 0x1159;
case 0xc000a2a3: return 0x115a;
case 0xc000a2a4: return 0x115b;
case 0xc000ce01: return 0x171;
case 0xc000ce02: return 0x172;
case 0xc000ce03: return 0x173;
case 0xc000ce04: return 0x174;
case 0xc000ce05: return 0x181;
case 0xc000cf00: return 0x166;
case 0xc000cf01: return 0x16a;
case 0xc000cf02: return 0x16b;
case 0xc000cf03: return 0x16c;
case 0xc000cf06: return 0x177;
case 0xc000cf07: return 0x178;
case 0xc000cf08: return 0x179;
case 0xc000cf09: return 0x17a;
case 0xc000cf0a: return 0x17b;
case 0xc000cf0b: return 0x17c;
case 0xc000cf0c: return 0x17d;
case 0xc000cf0d: return 0x17e;
case 0xc000cf0e: return 0x17f;
case 0xc000cf0f: return 0x182;
case 0xc000cf10: return 0x183;
case 0xc000cf11: return 0x184;
case 0xc000cf12: return 0x185;
case 0xc000cf13: return 0x186;
case 0xc000cf14: return 0x187;
case 0xc000cf15: return 0x188;
case 0xc000cf16: return 0x189;
case 0xc000cf17: return 0x18a;
case 0xc000cf18: return 0x18b;
case 0xc000cf19: return 0x18c;
case 0xc000cf1a: return 0x18d;
case 0xc000cf1b: return 0x18e;
    }
    return static_cast<win32::DWORD>(-1);
  }
  //! Construct from a NT error code
  static _base::string_ref _make_string_ref(win32::NTSTATUS c) noexcept
  {
    wchar_t buffer[32768];
    static win32::HMODULE ntdll = win32::GetModuleHandleW(L"NTDLL.DLL");
    win32::DWORD wlen = win32::FormatMessageW(0x00000800 /*FORMAT_MESSAGE_FROM_HMODULE*/ | 0x00001000 /*FORMAT_MESSAGE_FROM_SYSTEM*/ | 0x00000200 /*FORMAT_MESSAGE_IGNORE_INSERTS*/, ntdll, c, (1 << 10) /*MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT)*/, buffer, 32768, nullptr);
    size_t allocation = wlen + (wlen >> 1);
    win32::DWORD bytes;
    if(wlen == 0)
    {
      return _base::string_ref("failed to get message from system");
    }
    for(;;)
    {
      auto *p = static_cast<char *>(malloc(allocation)); // NOLINT
      if(p == nullptr)
      {
        return _base::string_ref("failed to get message from system");
      }
      bytes = win32::WideCharToMultiByte(65001 /*CP_UTF8*/, 0, buffer, (int) (wlen + 1), p, (int) allocation, nullptr, nullptr);
      if(bytes != 0)
      {
        char *end = strchr(p, 0);
        while(end[-1] == 10 || end[-1] == 13)
        {
          --end;
        }
        *end = 0; // NOLINT
        return _base::atomic_refcounted_string_ref(p, end - p);
      }
      free(p); // NOLINT
      if(win32::GetLastError() == 0x7a /*ERROR_INSUFFICIENT_BUFFER*/)
      {
        allocation += allocation >> 2;
        continue;
      }
      return _base::string_ref("failed to get message from system");
    }
  }
public:
  //! The value type of the NT code, which is a `win32::NTSTATUS`
  using value_type = win32::NTSTATUS;
  using _base::string_ref;
public:
  //! Default constructor
  constexpr explicit _nt_code_domain(typename _base::unique_id_type id = 0x93f3b4487e4af25b) noexcept
      : _base(id)
  {
  }
  _nt_code_domain(const _nt_code_domain &) = default;
  _nt_code_domain(_nt_code_domain &&) = default;
  _nt_code_domain &operator=(const _nt_code_domain &) = default;
  _nt_code_domain &operator=(_nt_code_domain &&) = default;
  ~_nt_code_domain() = default;
  //! Constexpr singleton getter. Returns the constexpr nt_code_domain variable.
  static inline constexpr const _nt_code_domain &get();
  virtual string_ref name() const noexcept override { return string_ref("NT domain"); } // NOLINT
protected:
  virtual bool _do_failure(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this);
    return static_cast<const nt_code &>(code).value() < 0; // NOLINT
  }
  virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept override // NOLINT
  {
    assert(code1.domain() == *this);
    const auto &c1 = static_cast<const nt_code &>(code1); // NOLINT
    if(code2.domain() == *this)
    {
      const auto &c2 = static_cast<const nt_code &>(code2); // NOLINT
      return c1.value() == c2.value();
    }
    if(code2.domain() == generic_code_domain)
    {
      const auto &c2 = static_cast<const generic_code &>(code2); // NOLINT
      if(static_cast<int>(c2.value()) == _nt_code_to_errno(c1.value()))
      {
        return true;
      }
    }
    if(code2.domain() == win32_code_domain)
    {
      const auto &c2 = static_cast<const win32_code &>(code2); // NOLINT
      if(c2.value() == _nt_code_to_win32_code(c1.value()))
      {
        return true;
      }
    }
    return false;
  }
  virtual generic_code _generic_code(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this);
    const auto &c = static_cast<const nt_code &>(code); // NOLINT
    return generic_code(static_cast<errc>(_nt_code_to_errno(c.value())));
  }
  virtual string_ref _do_message(const status_code<void> &code) const noexcept override // NOLINT
  {
    assert(code.domain() == *this);
    const auto &c = static_cast<const nt_code &>(code); // NOLINT
    return _make_string_ref(c.value());
  }
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || 0L
  SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const override // NOLINT
  {
    assert(code.domain() == *this);
    const auto &c = static_cast<const nt_code &>(code); // NOLINT
    throw status_error<_nt_code_domain>(c);
  }
#endif
};
//! (Windows only) A constexpr source variable for the NT code domain, which is that of NT kernel functions. Returned by `_nt_code_domain::get()`.
constexpr _nt_code_domain nt_code_domain;
inline constexpr const _nt_code_domain &_nt_code_domain::get()
{
  return nt_code_domain;
}
SYSTEM_ERROR2_NAMESPACE_END
#endif
// NOT "com_code.hpp"
#endif
SYSTEM_ERROR2_NAMESPACE_BEGIN
/*! An erased-mutable status code suitably large for all the system codes
which can be returned on this system.

For Windows, these might be:

    - `com_code` (`HRESULT`)  [you need to include "com_code.hpp" explicitly for this]
    - `nt_code` (`LONG`)
    - `win32_code` (`DWORD`)

For POSIX, `posix_code` is possible.

You are guaranteed that `system_code` can be transported by the compiler
in exactly two CPU registers.
*/
using system_code = status_code<erased<intptr_t>>;
#ifndef NDEBUG
static_assert(sizeof(system_code) == 2 * sizeof(void *), "system_code is not exactly two pointers in size!");
static_assert(traits::is_move_bitcopying<system_code>::value, "system_code is not move bitcopying!");
#endif
SYSTEM_ERROR2_NAMESPACE_END
#endif
SYSTEM_ERROR2_NAMESPACE_BEGIN
/*! An erased `system_code` which is always a failure. The closest equivalent to
`std::error_code`, except it cannot be null and cannot be modified.

This refines `system_code` into an `error` object meeting the requirements of
[P0709 Zero-overhead deterministic exceptions](https://wg21.link/P0709).

Differences from `system_code`:

- Always a failure (this is checked at construction, and if not the case,
the program is terminated as this is a logic error)
- No default construction.
- No empty state possible.
- Is immutable.

As with `system_code`, it remains guaranteed to be two CPU registers in size,
and move bitcopying.
*/
using error = errored_status_code<erased<system_code::value_type>>;
#ifndef NDEBUG
static_assert(sizeof(error) == 2 * sizeof(void *), "error is not exactly two pointers in size!");
static_assert(traits::is_move_bitcopying<error>::value, "error is not move bitcopying!");
#endif
SYSTEM_ERROR2_NAMESPACE_END
#endif
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace trait
{
  namespace detail
  {
    // Shortcut this for lower build impact. Used to tell outcome's converting constructors
    // that they can do E => EC or E => EP as necessary.
    template <class DomainType> struct _is_error_code_available<SYSTEM_ERROR2_NAMESPACE::status_code<DomainType>>
    {
      static constexpr bool value = true;
      using type = SYSTEM_ERROR2_NAMESPACE::status_code<DomainType>;
    };
    template <class DomainType> struct _is_error_code_available<SYSTEM_ERROR2_NAMESPACE::errored_status_code<DomainType>>
    {
      static constexpr bool value = true;
      using type = SYSTEM_ERROR2_NAMESPACE::errored_status_code<DomainType>;
    };
  } // namespace detail
  template <class DomainType> struct is_move_bitcopying<SYSTEM_ERROR2_NAMESPACE::status_code<DomainType>>
  {
    static constexpr bool value = SYSTEM_ERROR2_NAMESPACE::traits::is_move_bitcopying<SYSTEM_ERROR2_NAMESPACE::status_code<DomainType>>::value;
  };
  template <class DomainType> struct is_move_bitcopying<SYSTEM_ERROR2_NAMESPACE::errored_status_code<DomainType>>
  {
    static constexpr bool value = SYSTEM_ERROR2_NAMESPACE::traits::is_move_bitcopying<SYSTEM_ERROR2_NAMESPACE::errored_status_code<DomainType>>::value;
  };
} // namespace trait
namespace detail
{
  // Customise _set_error_is_errno
  template <class State> constexpr inline void _set_error_is_errno(State &state, const SYSTEM_ERROR2_NAMESPACE::generic_code & /*unused*/)
  {
    state._status.set_have_error_is_errno(true);
  }
#ifndef SYSTEM_ERROR2_NOT_POSIX
  template <class State> constexpr inline void _set_error_is_errno(State &state, const SYSTEM_ERROR2_NAMESPACE::posix_code & /*unused*/)
  {
    state._status.set_have_error_is_errno(true);
  }
#endif
  template <class State> constexpr inline void _set_error_is_errno(State &state, const SYSTEM_ERROR2_NAMESPACE::errc & /*unused*/)
  {
    state._status.set_have_error_is_errno(true);
  }
} // namespace detail
namespace experimental
{
  using namespace SYSTEM_ERROR2_NAMESPACE;
  using OUTCOME_V2_NAMESPACE::failure;
  using OUTCOME_V2_NAMESPACE::success;
  namespace policy
  {
    using namespace OUTCOME_V2_NAMESPACE::policy;
    template <class T, class EC, class E> struct status_code_throw
    {
      static_assert(!std::is_same<T, T>::value,
                    "policy::status_code_throw not specialised for these types, did you use status_result<T, status_code<DomainType>, E>?");
    };
    template <class T, class DomainType> struct status_code_throw<T, status_code<DomainType>, void> : base
    {
      using _base = base;
      template <class Impl> static constexpr void wide_value_check(Impl &&self)
      {
        if(!base::_has_value(static_cast<Impl &&>(self)))
        {
          if(base::_has_error(static_cast<Impl &&>(self)))
          {
#ifdef __cpp_exceptions
            base::_error(static_cast<Impl &&>(self)).throw_exception();
#else
            OUTCOME_THROW_EXCEPTION("wide value check failed");
#endif
          }
        }
      }
      template <class Impl> static constexpr void wide_error_check(Impl &&self) { _base::narrow_error_check(static_cast<Impl &&>(self)); }
    };
    template <class T, class DomainType>
    struct status_code_throw<T, errored_status_code<DomainType>, void> : status_code_throw<T, status_code<DomainType>, void>
    {
      status_code_throw() = default;
      using status_code_throw<T, status_code<DomainType>, void>::status_code_throw;
    };
    template <class T, class EC>
    using default_status_result_policy = std::conditional_t< //
    std::is_void<EC>::value, //
    OUTCOME_V2_NAMESPACE::policy::terminate, //
    std::conditional_t<is_status_code<EC>::value || is_errored_status_code<EC>::value, //
                       status_code_throw<T, EC, void>, //
                       OUTCOME_V2_NAMESPACE::policy::fail_to_compile_observers //
                       >>;
  } // namespace policy
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S = errored_status_code<erased<typename system_code::value_type>>,
            class NoValuePolicy = policy::default_status_result_policy<R, S>> //
  using status_result = basic_result<R, S, NoValuePolicy>;
} // namespace experimental
OUTCOME_V2_NAMESPACE_END
#endif
// Boost.Outcome #include "boost/exception_ptr.hpp"
SYSTEM_ERROR2_NAMESPACE_BEGIN
template <class DomainType> inline std::exception_ptr basic_outcome_failure_exception_from_error(const status_code<DomainType> &sc)
{
  (void) sc;
#ifdef __cpp_exceptions
  try
  {
    sc.throw_exception();
  }
  catch(...)
  {
    return std::current_exception();
  }
#endif
  return {};
}
SYSTEM_ERROR2_NAMESPACE_END
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace experimental
{
  namespace policy
  {
    template <class T, class EC, class E>
    using default_status_outcome_policy = std::conditional_t< //
    std::is_void<EC>::value && std::is_void<E>::value, //
    OUTCOME_V2_NAMESPACE::policy::terminate, //
    std::conditional_t<(is_status_code<EC>::value || is_errored_status_code<EC>::value) &&
                       (std::is_void<E>::value || OUTCOME_V2_NAMESPACE::trait::is_exception_ptr_available<E>::value), //
                       status_code_throw<T, EC, E>, //
                       OUTCOME_V2_NAMESPACE::policy::fail_to_compile_observers //
                       >>;
  } // namespace policy
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S = errored_status_code<erased<typename system_code::value_type>>, class P = std::exception_ptr,
            class NoValuePolicy = policy::default_status_outcome_policy<R, S, P>> //
  using status_outcome = basic_outcome<R, S, P, NoValuePolicy>;
  namespace policy
  {
    template <class T, class DomainType, class E> struct status_code_throw<T, status_code<DomainType>, E> : base
    {
      using _base = base;
      template <class Impl> static constexpr void wide_value_check(Impl &&self)
      {
        if(!base::_has_value(static_cast<Impl &&>(self)))
        {
          if(base::_has_exception(static_cast<Impl &&>(self)))
          {
            OUTCOME_V2_NAMESPACE::policy::detail::_rethrow_exception<trait::is_exception_ptr_available<E>::value>(
            base::_exception<T, status_code<DomainType>, E, status_code_throw>(static_cast<Impl &&>(self))); // NOLINT
          }
          if(base::_has_error(static_cast<Impl &&>(self)))
          {
#ifdef __cpp_exceptions
            base::_error(static_cast<Impl &&>(self)).throw_exception();
#else
            OUTCOME_THROW_EXCEPTION("wide value check failed");
#endif
          }
        }
      }
      template <class Impl> static constexpr void wide_error_check(Impl &&self) { _base::narrow_error_check(static_cast<Impl &&>(self)); }
      template <class Impl> static constexpr void wide_exception_check(Impl &&self) { _base::narrow_exception_check(static_cast<Impl &&>(self)); }
    };
    template <class T, class DomainType, class E>
    struct status_code_throw<T, errored_status_code<DomainType>, E> : status_code_throw<T, status_code<DomainType>, E>
    {
      status_code_throw() = default;
      using status_code_throw<T, status_code<DomainType>, E>::status_code_throw;
    };
  } // namespace policy
} // namespace experimental
OUTCOME_V2_NAMESPACE_END
#endif
/* Try operation macros
(C) 2017-2021 Niall Douglas <http://www.nedproductions.biz/> (20 commits)
File Created: July 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_TRY_HPP
#define OUTCOME_TRY_HPP
OUTCOME_V2_NAMESPACE_BEGIN
namespace detail
{
  struct has_value_overload
  {
  };
  struct as_failure_overload
  {
  };
  struct assume_error_overload
  {
  };
  struct error_overload
  {
  };
  struct assume_value_overload
  {
  };
  struct value_overload
  {
  };
  //#ifdef __APPLE__
  //  OUTCOME_TEMPLATE(class T, class R = decltype(std::declval<T>()._xcode_workaround_as_failure()))
  //#else
  OUTCOME_TEMPLATE(class T, class R = decltype(std::declval<T>().as_failure()))
  //#endif
  OUTCOME_TREQUIRES(OUTCOME_TPRED(OUTCOME_V2_NAMESPACE::is_failure_type<R>))
  constexpr inline bool has_as_failure(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_as_failure(...) { return false; }
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().assume_error()))
  constexpr inline bool has_assume_error(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_assume_error(...) { return false; }
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().error()))
  constexpr inline bool has_error(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_error(...) { return false; }
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().assume_value()))
  constexpr inline bool has_assume_value(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_assume_value(...) { return false; }
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().value()))
  constexpr inline bool has_value(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_value(...) { return false; }
} // namespace detail
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().has_value()))
constexpr inline bool try_operation_has_value(T &&v, detail::has_value_overload = {})
{
  return v.has_value();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::has_as_failure<T>(5)))
constexpr inline decltype(auto) try_operation_return_as(T &&v, detail::as_failure_overload = {})
{
  return static_cast<T &&>(v).as_failure();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(!detail::has_as_failure<T>(5) && detail::has_assume_error<T>(5)))
constexpr inline decltype(auto) try_operation_return_as(T &&v, detail::assume_error_overload = {})
{
  return failure(static_cast<T &&>(v).assume_error());
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(!detail::has_as_failure<T>(5) && !detail::has_assume_error<T>(5) && detail::has_error<T>(5)))
constexpr inline decltype(auto) try_operation_return_as(T &&v, detail::error_overload = {})
{
  return failure(static_cast<T &&>(v).error());
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::has_assume_value<T>(5)))
constexpr inline decltype(auto) try_operation_extract_value(T &&v, detail::assume_value_overload = {})
{
  return static_cast<T &&>(v).assume_value();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(!detail::has_assume_value<T>(5) && detail::has_value<T>(5)))
constexpr inline decltype(auto) try_operation_extract_value(T &&v, detail::value_overload = {})
{
  return static_cast<T &&>(v).value();
}
OUTCOME_V2_NAMESPACE_END
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
#define OUTCOME_TRY_GLUE2(x, y) x##y
#define OUTCOME_TRY_GLUE(x, y) OUTCOME_TRY_GLUE2(x, y)
#define OUTCOME_TRY_UNIQUE_NAME OUTCOME_TRY_GLUE(_outcome_try_unique_name_temporary, __COUNTER__)
#define OUTCOME_TRY_RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, count, ...) count
#define OUTCOME_TRY_EXPAND_ARGS(args) OUTCOME_TRY_RETURN_ARG_COUNT args
#define OUTCOME_TRY_COUNT_ARGS_MAX8(...) OUTCOME_TRY_EXPAND_ARGS((__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define OUTCOME_TRY_OVERLOAD_MACRO2(name, count) name##count
#define OUTCOME_TRY_OVERLOAD_MACRO1(name, count) OUTCOME_TRY_OVERLOAD_MACRO2(name, count)
#define OUTCOME_TRY_OVERLOAD_MACRO(name, count) OUTCOME_TRY_OVERLOAD_MACRO1(name, count)
#define OUTCOME_TRY_OVERLOAD_GLUE(x, y) x y
#define OUTCOME_TRY_CALL_OVERLOAD(name, ...) OUTCOME_TRY_OVERLOAD_GLUE(OUTCOME_TRY_OVERLOAD_MACRO(name, OUTCOME_TRY_COUNT_ARGS_MAX8(__VA_ARGS__)), (__VA_ARGS__))
#define _OUTCOME_TRY_RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, count, ...) count
#define _OUTCOME_TRY_EXPAND_ARGS(args) _OUTCOME_TRY_RETURN_ARG_COUNT args
#define _OUTCOME_TRY_COUNT_ARGS_MAX8(...) _OUTCOME_TRY_EXPAND_ARGS((__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define _OUTCOME_TRY_OVERLOAD_MACRO2(name, count) name##count
#define _OUTCOME_TRY_OVERLOAD_MACRO1(name, count) _OUTCOME_TRY_OVERLOAD_MACRO2(name, count)
#define _OUTCOME_TRY_OVERLOAD_MACRO(name, count) _OUTCOME_TRY_OVERLOAD_MACRO1(name, count)
#define _OUTCOME_TRY_OVERLOAD_GLUE(x, y) x y
#define _OUTCOME_TRY_CALL_OVERLOAD(name, ...) _OUTCOME_TRY_OVERLOAD_GLUE(_OUTCOME_TRY_OVERLOAD_MACRO(name, _OUTCOME_TRY_COUNT_ARGS_MAX8(__VA_ARGS__)), (__VA_ARGS__))
#ifndef OUTCOME_TRY_LIKELY_IF
#if (__cplusplus >= 202000L || _HAS_CXX20) && (!defined(__clang__) || __clang_major__ >= 12)
#define OUTCOME_TRY_LIKELY_IF(...) if(__VA_ARGS__) [[likely]]
#elif defined(__clang__) || defined(__GNUC__)
#define OUTCOME_TRY_LIKELY_IF(...) if(__builtin_expect(!!(__VA_ARGS__), true))
#else
#define OUTCOME_TRY_LIKELY_IF(...) if(__VA_ARGS__)
#endif
#endif
#define OUTCOME_TRYV2_UNIQUE_STORAGE_UNPACK(...) __VA_ARGS__
#define OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE3(unique, ...) auto unique = (__VA_ARGS__)
#define OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE2(x) x
#define OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE(unique, x, ...) OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE2(OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE3(unique, __VA_ARGS__))
#define OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED3(unique, x, y, ...) x unique = (__VA_ARGS__)
#define OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED2(x) x
#define OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED(unique, ...) OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED2(OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED3(unique, __VA_ARGS__))
#define OUTCOME_TRYV2_UNIQUE_STORAGE1(...) OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE
#define OUTCOME_TRYV2_UNIQUE_STORAGE2(...) OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED
#define OUTCOME_TRYV2_UNIQUE_STORAGE(unique, spec, ...) _OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRYV2_UNIQUE_STORAGE, OUTCOME_TRYV2_UNIQUE_STORAGE_UNPACK spec) (unique, OUTCOME_TRYV2_UNIQUE_STORAGE_UNPACK spec, __VA_ARGS__)
// Use if(!expr); else as some compilers assume else clauses are always unlikely
#define OUTCOME_TRYV2_SUCCESS_LIKELY(unique, retstmt, spec, ...) OUTCOME_TRYV2_UNIQUE_STORAGE(unique, spec, __VA_ARGS__); OUTCOME_TRY_LIKELY_IF(::OUTCOME_V2_NAMESPACE::try_operation_has_value(unique)); else retstmt ::OUTCOME_V2_NAMESPACE::try_operation_return_as(static_cast<decltype(unique) &&>(unique))
#define OUTCOME_TRYV3_FAILURE_LIKELY(unique, retstmt, spec, ...) OUTCOME_TRYV2_UNIQUE_STORAGE(unique, spec, __VA_ARGS__); OUTCOME_TRY_LIKELY_IF(!OUTCOME_V2_NAMESPACE::try_operation_has_value(unique)) retstmt ::OUTCOME_V2_NAMESPACE::try_operation_return_as(static_cast<decltype(unique) &&>(unique))
#define OUTCOME_TRY2_VAR_SECOND2(x, var) var
#define OUTCOME_TRY2_VAR_SECOND3(x, y, ...) x y
#define OUTCOME_TRY2_VAR(spec) _OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRY2_VAR_SECOND, OUTCOME_TRYV2_UNIQUE_STORAGE_UNPACK spec, spec)
#define OUTCOME_TRY2_SUCCESS_LIKELY(unique, retstmt, var, ...) OUTCOME_TRYV2_SUCCESS_LIKELY(unique, retstmt, var, __VA_ARGS__); OUTCOME_TRY2_VAR(var) = ::OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique))
#define OUTCOME_TRY2_FAILURE_LIKELY(unique, retstmt, var, ...) OUTCOME_TRYV3_FAILURE_LIKELY(unique, retstmt, var, __VA_ARGS__); OUTCOME_TRY2_VAR(var) = ::OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique))
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYV(...) OUTCOME_TRYV2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, deduce, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYV_FAILURE_LIKELY(...) OUTCOME_TRYV3_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, deduce, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYV(...) OUTCOME_TRYV2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, deduce, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYV_FAILURE_LIKELY(...) OUTCOME_TRYV3_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, deduce, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYV2(s, ...) OUTCOME_TRYV2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, (s,), __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYV2_FAILURE_LIKELY(s, ...) OUTCOME_TRYV3_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, (s,), __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYV2(s, ...) OUTCOME_TRYV2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, (s,), __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYV2_FAILURE_LIKELY(s, ...) OUTCOME_TRYV3_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, s(,), __VA_ARGS__)
#if defined(__GNUC__) || defined(__clang__)
#define OUTCOME_TRYX2(unique, retstmt, ...) ({ OUTCOME_TRYV2_SUCCESS_LIKELY(unique, retstmt, deduce, __VA_ARGS__); ::OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique)); })
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYX(...) OUTCOME_TRYX2(OUTCOME_TRY_UNIQUE_NAME, return, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYX(...) OUTCOME_TRYX2(OUTCOME_TRY_UNIQUE_NAME, co_return, __VA_ARGS__)
#endif
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYA(v, ...) OUTCOME_TRY2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYA_FAILURE_LIKELY(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYA(v, ...) OUTCOME_TRY2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYA_FAILURE_LIKELY(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, v, __VA_ARGS__)
#define OUTCOME_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME_TRYA(a, b, c, d, e, f, g, h)
#define OUTCOME_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME_TRYA(a, b, c, d, e, f, g)
#define OUTCOME_TRY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME_TRYA(a, b, c, d, e, f)
#define OUTCOME_TRY_INVOKE_TRY5(a, b, c, d, e) OUTCOME_TRYA(a, b, c, d, e)
#define OUTCOME_TRY_INVOKE_TRY4(a, b, c, d) OUTCOME_TRYA(a, b, c, d)
#define OUTCOME_TRY_INVOKE_TRY3(a, b, c) OUTCOME_TRYA(a, b, c)
#define OUTCOME_TRY_INVOKE_TRY2(a, b) OUTCOME_TRYA(a, b)
#define OUTCOME_TRY_INVOKE_TRY1(a) OUTCOME_TRYV(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) OUTCOME_TRYA_FAILURE_LIKELY(a, b)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) OUTCOME_TRYV_FAILURE_LIKELY(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRY_FAILURE_LIKELY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME_CO_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME_CO_TRYA(a, b, c, d, e, f, g, h)
#define OUTCOME_CO_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME_CO_TRYA(a, b, c, d, e, f, g)
#define OUTCOME_CO_TRY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME_CO_TRYA(a, b, c, d, e, f)
#define OUTCOME_CO_TRY_INVOKE_TRY5(a, b, c, d, e) OUTCOME_CO_TRYA(a, b, c, d, e)
#define OUTCOME_CO_TRY_INVOKE_TRY4(a, b, c, d) OUTCOME_CO_TRYA(a, b, c, d)
#define OUTCOME_CO_TRY_INVOKE_TRY3(a, b, c) OUTCOME_CO_TRYA(a, b, c)
#define OUTCOME_CO_TRY_INVOKE_TRY2(a, b) OUTCOME_CO_TRYA(a, b)
#define OUTCOME_CO_TRY_INVOKE_TRY1(a) OUTCOME_CO_TRYV(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_CO_TRY_INVOKE_TRY, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) OUTCOME_CO_TRYV_FAILURE_LIKELY(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRY_FAILURE_LIKELY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_TRYA(v, ...) OUTCOME_TRY2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, deduce, auto &&v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_TRYA_FAILURE_LIKELY(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, deduce, auto &&v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_CO_TRYA(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, deduce, auto &&v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_CO_TRYA_FAILURE_LIKELY(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_retrn, deduce, auto &&v, __VA_ARGS__)
#define OUTCOME21_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME21_TRYA(a, b, c, d, e, f, g, h)
#define OUTCOME21_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME21_TRYA(a, b, c, d, e, f, g)
#define OUTCOME21_TRY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME21_TRYA(a, b, c, d, e, f)
#define OUTCOME21_TRY_INVOKE_TRY5(a, b, c, d, e) OUTCOME21_TRYA(a, b, c, d, e)
#define OUTCOME21_TRY_INVOKE_TRY4(a, b, c, d) OUTCOME21_TRYA(a, b, c, d)
#define OUTCOME21_TRY_INVOKE_TRY3(a, b, c) OUTCOME21_TRYA(a, b, c)
#define OUTCOME21_TRY_INVOKE_TRY2(a, b) OUTCOME21_TRYA(a, b)
#define OUTCOME21_TRY_INVOKE_TRY1(a) OUTCOME_TRYV(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME21_TRY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) OUTCOME21_TRYA_FAILURE_LIKELY(a, b)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) OUTCOME_TRYV_FAILURE_LIKELY(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_TRY_FAILURE_LIKELY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME21_CO_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME21_CO_TRYA(a, b, c, d, e, f, g, h)
#define OUTCOME21_CO_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME21_CO_TRYA(a, b, c, d, e, f, g)
#define OUTCOME21_CO_TRY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME21_CO_TRYA(a, b, c, d, e, f)
#define OUTCOME21_CO_TRY_INVOKE_TRY5(a, b, c, d, e) OUTCOME21_CO_TRYA(a, b, c, d, e)
#define OUTCOME21_CO_TRY_INVOKE_TRY4(a, b, c, d) OUTCOME21_CO_TRYA(a, b, c, d)
#define OUTCOME21_CO_TRY_INVOKE_TRY3(a, b, c) OUTCOME21_CO_TRYA(a, b, c)
#define OUTCOME21_CO_TRY_INVOKE_TRY2(a, b) OUTCOME21_CO_TRYA(a, b)
#define OUTCOME21_CO_TRY_INVOKE_TRY1(a) OUTCOME_CO_TRYV(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_CO_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME21_CO_TRY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) OUTCOME_CO_TRYV_FAILURE_LIKELY(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_CO_TRY_FAILURE_LIKELY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif
#endif
