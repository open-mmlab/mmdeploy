// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)

// spdlog main header file.
// see example.cpp for usage example

#ifndef SPDLOG_H
#define SPDLOG_H

#pragma once

#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/version.h>
#include <spdlog/details/synchronous_factory.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace spdlog {

// API for using default logger (stdout_color_mt),
// e.g: spdlog::info("Message {}", 1);
//
// The default logger object can be accessed using the spdlog::default_logger():
// For example, to add another sink to it:
// spdlog::default_logger()->sinks().push_back(some_sink);
//
// The default logger can replaced using spdlog::set_default_logger(new_logger).
// For example, to replace it with a file logger.
//
// IMPORTANT:
// The default API is thread safe (for _mt loggers), but:
// set_default_logger() *should not* be used concurrently with the default API.
// e.g do not call set_default_logger() from one thread while calling spdlog::info() from another.

SPDLOG_API std::shared_ptr<spdlog::logger> &default_logger();
SPDLOG_API void set_default_logger(std::shared_ptr<spdlog::logger> default_logger);

template<typename... Args>
inline void log(source_loc source, level::level_enum lvl, format_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->log(source, lvl, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void log(level::level_enum lvl, format_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->log(source_loc{}, lvl, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void trace(format_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->trace(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void debug(format_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->debug(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void info(format_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->info(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void warn(format_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->warn(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void error(format_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->error(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void critical(format_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->critical(fmt, std::forward<Args>(args)...);
}

template<typename T>
inline void log(source_loc source, level::level_enum lvl, const T &msg)
{
    default_logger()->log(source, lvl, msg);
}

template<typename T>
inline void log(level::level_enum lvl, const T &msg)
{
    default_logger()->log(lvl, msg);
}

#ifdef SPDLOG_WCHAR_TO_UTF8_SUPPORT
template<typename... Args>
inline void log(source_loc source, level::level_enum lvl, wformat_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->log(source, lvl, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void log(level::level_enum lvl, wformat_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->log(source_loc{}, lvl, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void trace(wformat_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->trace(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void debug(wformat_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->debug(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void info(wformat_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->info(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void warn(wformat_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->warn(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void error(wformat_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->error(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void critical(wformat_string_t<Args...> fmt, Args &&... args)
{
    default_logger()->critical(fmt, std::forward<Args>(args)...);
}
#endif

template<typename T>
inline void trace(const T &msg)
{
    default_logger()->trace(msg);
}

template<typename T>
inline void debug(const T &msg)
{
    default_logger()->debug(msg);
}

template<typename T>
inline void info(const T &msg)
{
    default_logger()->info(msg);
}

template<typename T>
inline void warn(const T &msg)
{
    default_logger()->warn(msg);
}

template<typename T>
inline void error(const T &msg)
{
    default_logger()->error(msg);
}

template<typename T>
inline void critical(const T &msg)
{
    default_logger()->critical(msg);
}

} // namespace spdlog

//
// enable/disable log calls at compile time according to global level.
//
// define SPDLOG_ACTIVE_LEVEL to one of those (before including spdlog.h):
// SPDLOG_LEVEL_TRACE,
// SPDLOG_LEVEL_DEBUG,
// SPDLOG_LEVEL_INFO,
// SPDLOG_LEVEL_WARN,
// SPDLOG_LEVEL_ERROR,
// SPDLOG_LEVEL_CRITICAL,
// SPDLOG_LEVEL_OFF
//

#define SPDLOG_LOGGER_CALL(logger, level, ...) (logger)->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, level, __VA_ARGS__)

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
#    define SPDLOG_LOGGER_TRACE(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::trace, __VA_ARGS__)
#    define SPDLOG_TRACE(...) SPDLOG_LOGGER_TRACE(spdlog::default_logger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_TRACE(logger, ...) (void)0
#    define SPDLOG_TRACE(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
#    define SPDLOG_LOGGER_DEBUG(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::debug, __VA_ARGS__)
#    define SPDLOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(spdlog::default_logger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_DEBUG(logger, ...) (void)0
#    define SPDLOG_DEBUG(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_INFO
#    define SPDLOG_LOGGER_INFO(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::info, __VA_ARGS__)
#    define SPDLOG_INFO(...) SPDLOG_LOGGER_INFO(spdlog::default_logger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_INFO(logger, ...) (void)0
#    define SPDLOG_INFO(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_WARN
#    define SPDLOG_LOGGER_WARN(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::warn, __VA_ARGS__)
#    define SPDLOG_WARN(...) SPDLOG_LOGGER_WARN(spdlog::default_logger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_WARN(logger, ...) (void)0
#    define SPDLOG_WARN(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_ERROR
#    define SPDLOG_LOGGER_ERROR(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::err, __VA_ARGS__)
#    define SPDLOG_ERROR(...) SPDLOG_LOGGER_ERROR(spdlog::default_logger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_ERROR(logger, ...) (void)0
#    define SPDLOG_ERROR(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_CRITICAL
#    define SPDLOG_LOGGER_CRITICAL(logger, ...) SPDLOG_LOGGER_CALL(logger, spdlog::level::critical, __VA_ARGS__)
#    define SPDLOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(spdlog::default_logger(), __VA_ARGS__)
#else
#    define SPDLOG_LOGGER_CRITICAL(logger, ...) (void)0
#    define SPDLOG_CRITICAL(...) (void)0
#endif

#ifdef SPDLOG_HEADER_ONLY
#    include "spdlog-inl.h"
#endif

#endif // SPDLOG_H
