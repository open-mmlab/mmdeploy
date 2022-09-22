// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/status_code.h"

namespace mmdeploy {

class Platform;
class Device;
class Stream;
class Event;
class Allocator;
class Buffer;
class Kernel;

class PlatformImpl;
class StreamImpl;
class EventImpl;
class AllocatorImpl;
class BufferImpl;
class KernelImpl;

template <typename T>
using optional = std::optional<T>;

class DeviceId {
 public:
  using ValueType = int32_t;
  constexpr explicit DeviceId(ValueType value) : value_(value) {}
  constexpr operator ValueType() const { return value_; }  // NOLINT
  constexpr ValueType get() const { return value_; }

 private:
  ValueType value_;
};

class PlatformId {
 public:
  using ValueType = int32_t;
  constexpr explicit PlatformId(ValueType value) : value_(value) {}
  constexpr operator ValueType() const { return value_; }  // NOLINT
  constexpr ValueType get() const { return value_; }

 private:
  ValueType value_;
};

class Device {
 public:
  constexpr Device() : platform_id_(-1), device_id_(-1) {}

  constexpr explicit Device(DeviceId device_id, PlatformId platform_id = PlatformId(-1))
      : Device(platform_id.get(), device_id.get()) {}

  constexpr explicit Device(PlatformId platform_id, DeviceId device_id = DeviceId(-1))
      : Device(platform_id.get(), device_id.get()) {}

  constexpr explicit Device(int platform_id, int device_id = 0)
      : platform_id_(platform_id), device_id_(device_id) {}

  MMDEPLOY_API explicit Device(const char* platform_name, int device_id = 0);

  constexpr int device_id() const noexcept { return device_id_; }

  constexpr int platform_id() const noexcept { return platform_id_; }

  constexpr bool is_host() const noexcept { return platform_id() == 0; }

  constexpr bool is_device() const noexcept { return platform_id() > 0; }

  constexpr bool operator==(const Device& other) const noexcept {
    return platform_id_ == other.platform_id_ && device_id_ == other.device_id_;
  }

  constexpr bool operator!=(const Device& other) const noexcept { return !(*this == other); }

  constexpr explicit operator bool() const noexcept { return platform_id_ >= 0 && device_id_ >= 0; }

  constexpr operator DeviceId() const noexcept {  // NOLINT
    return DeviceId(device_id_);
  }

  constexpr operator PlatformId() const noexcept {  // NOLINT
    return PlatformId(platform_id_);
  }

 private:
  int platform_id_{0};
  int device_id_{0};
};

enum class MemcpyKind : int { HtoD, DtoH, DtoD };

class MMDEPLOY_API Platform {
 public:
  // throws if not found
  explicit Platform(const char* platform_name);

  // throws if not found
  explicit Platform(int platform_id);

  // -1 if invalid
  int GetPlatformId() const;

  // "" if invalid
  const char* GetPlatformName() const;

  bool operator==(const Platform& other) { return impl_ == other.impl_; }

  bool operator!=(const Platform& other) { return !(*this == other); }

  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

 private:
  explicit Platform(std::shared_ptr<PlatformImpl> impl) : impl_(std::move(impl)) {}

 private:
  friend class PlatformRegistry;
  friend class Access;
  std::shared_ptr<PlatformImpl> impl_;
};

Platform GetPlatform(int platform_id);

Platform GetPlatform(const char* platform_name);

class MMDEPLOY_API Stream {
 public:
  Stream() = default;

  explicit Stream(Device device, uint64_t flags = 0);

  explicit Stream(Device device, void* native, uint64_t flags = 0);

  explicit Stream(Device device, std::shared_ptr<void> native, uint64_t flags = 0);

  Device GetDevice() const;

  Result<void> Query();

  Result<void> Wait();

  Result<void> DependsOn(Event& event);

  Result<void> Submit(Kernel& kernel);

  void* GetNative(ErrorCode* ec = nullptr);

  Result<void> Copy(const Buffer& src, Buffer& dst, size_t size = -1, size_t src_offset = 0,
                    size_t dst_offset = 0);

  Result<void> Copy(const void* host_ptr, Buffer& dst, size_t size = -1, size_t dst_offset = 0);

  Result<void> Copy(const Buffer& src, void* host_ptr, size_t size = -1, size_t src_offset = 0);

  Result<void> Fill(const Buffer& dst, void* pattern, size_t pattern_size, size_t size = -1,
                    size_t offset = 0);

  bool operator==(const Stream& other) const { return impl_ == other.impl_; }

  bool operator!=(const Stream& other) const { return !(*this == other); }

  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

  static Stream GetDefault(Device device);

 private:
  explicit Stream(std::shared_ptr<StreamImpl> impl) : impl_(std::move(impl)) {}

 private:
  friend class Access;

  std::shared_ptr<StreamImpl> impl_;
};

template <typename T>
T GetNative(Stream& stream, ErrorCode* ec = nullptr) {
  return reinterpret_cast<T>(stream.GetNative(ec));
}

class MMDEPLOY_API Event {
 public:
  Event() = default;

  explicit Event(Device device, uint64_t flags = 0);

  explicit Event(Device device, void* native, uint64_t flags = 0);

  explicit Event(Device device, std::shared_ptr<void> native, uint64_t flags = 0);

  Device GetDevice();

  Result<void> Query();

  Result<void> Wait();

  Result<void> Record(Stream& stream);

  void* GetNative(ErrorCode* ec = nullptr);

  bool operator==(const Event& other) const { return impl_ == other.impl_; }

  bool operator!=(const Event& other) const { return !(*this == other); }

  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

 private:
  explicit Event(std::shared_ptr<EventImpl> impl) : impl_(std::move(impl)) {}

 private:
  friend class Access;
  std::shared_ptr<EventImpl> impl_;
};

template <typename T>
T GetNative(Event& event, ErrorCode* ec = nullptr) {
  return reinterpret_cast<T>(event.GetNative(ec));
}

class MMDEPLOY_API Kernel {
 public:
  Kernel() = default;
  explicit Kernel(std::shared_ptr<KernelImpl> impl) : impl_(std::move(impl)) {}

  Device GetDevice() const;

  void* GetNative(ErrorCode* ec = nullptr);

  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

 private:
  std::shared_ptr<KernelImpl> impl_;
};

template <typename T>
T GetNative(Kernel& kernel, ErrorCode* ec = nullptr) {
  return reinterpret_cast<T>(kernel.GetNative(ec));
}

class MMDEPLOY_API Allocator {
  friend class Access;

 public:
  Allocator() = default;

  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

 private:
  explicit Allocator(std::shared_ptr<AllocatorImpl> impl) : impl_(std::move(impl)) {}
  std::shared_ptr<AllocatorImpl> impl_;
};

class MMDEPLOY_API Buffer {
 public:
  Buffer() = default;

  Buffer(Device device, size_t size, size_t alignment = 1, uint64_t flags = 0)
      : Buffer(device, size, Allocator{}, alignment, flags) {}

  Buffer(Device device, size_t size, Allocator allocator, size_t alignment = 1, uint64_t flags = 0);

  Buffer(Device device, size_t size, void* native, uint64_t flags = 0);

  Buffer(Device device, size_t size, std::shared_ptr<void> native, uint64_t flags = 0);
  // create sub-buffer
  Buffer(Buffer& buffer, size_t offset, size_t size, uint64_t flags = 0);

  size_t GetSize(ErrorCode* ec = nullptr) const;

  //  bool IsSubBuffer(ErrorCode* ec = nullptr);

  void* GetNative(ErrorCode* ec = nullptr) const;

  Device GetDevice() const;

  Allocator GetAllocator() const;

  bool operator==(const Buffer& other) const { return impl_ == other.impl_; }

  bool operator!=(const Buffer& other) const { return !(*this == other); }

  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

 private:
  explicit Buffer(std::shared_ptr<BufferImpl> impl) : impl_(std::move(impl)) {}

 private:
  friend class Access;
  std::shared_ptr<BufferImpl> impl_;
};

template <typename T>
T GetNative(Buffer& buffer, ErrorCode* ec = nullptr) {
  return reinterpret_cast<T>(buffer.GetNative(ec));
}

template <typename T>
T GetNative(const Buffer& buffer, ErrorCode* ec = nullptr) {
  return reinterpret_cast<T>(buffer.GetNative(ec));
}

class MMDEPLOY_API PlatformRegistry {
 public:
  using Creator = std::function<std::shared_ptr<PlatformImpl>()>;

  int Register(Creator creator);

  int AddAlias(const char* name, const char* target);

  int GetPlatform(const char* name, Platform* platform);

  int GetPlatform(int id, Platform* platform);

  int GetPlatformId(const char* name);

  PlatformImpl* GetPlatformImpl(PlatformId id);

 private:
  int GetNextId();

  bool IsAvailable(int id);

 private:
  struct Entry {
    std::string name;
    int id;
    Platform platform;
  };
  std::vector<Entry> entries_;
  std::vector<std::pair<std::string, std::string>> aliases_;
};

MMDEPLOY_API PlatformRegistry& gPlatformRegistry();

}  // namespace mmdeploy
