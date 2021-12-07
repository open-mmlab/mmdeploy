# Copyright (c) OpenMMLab. All rights reserved.
find_package(Boost 1.65 COMPONENTS stacktrace_backtrace)
if (Boost_FOUND)
    list(APPEND THIRDPARTY_DEFINITIONS -DUSE_BOOST_STACKTRACE=1)
    list(APPEND THIRDPARTY_LIBS Boost::stacktrace_backtrace)
endif()
