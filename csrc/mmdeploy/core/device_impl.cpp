// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/device_impl.h"

#include <cassert>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/logger.h"

namespace mmdeploy {

template <typename T>
T SetError(ErrorCode* ec, ErrorCode code, T ret) {
  if (ec) {
    *ec = code;
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Device

Device::Device(const char* platform_name, int device_id) {
  platform_id_ = gPlatformRegistry().GetPlatformId(platform_name);
  device_id_ = device_id;
}

//////////////////////////////////////////////////
/// Platform

int Platform::GetPlatformId() const {
  if (impl_) {
    return impl_->GetPlatformId();
  }
  return -1;
}

const char* Platform::GetPlatformName() const {
  if (impl_) {
    return impl_->GetPlatformName();
  }
  return "";
}

Platform::Platform(const char* platform_name) {
  if (-1 == gPlatformRegistry().GetPlatform(platform_name, this)) {
    throw_exception(eInvalidArgument);
  }
}

Platform::Platform(int platform_id) {
  if (-1 == gPlatformRegistry().GetPlatform(platform_id, this)) {
    throw_exception(eInvalidArgument);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Buffer

Buffer::Buffer(Device device, size_t size, Allocator allocator, size_t alignment, uint64_t flags) {
  if (auto p = GetPlatformImpl(device)) {
    impl_ = p->CreateBuffer(device);
    if (auto r = impl_->Init(size, std::move(allocator), alignment, flags); r.has_error()) {
      impl_.reset();
      r.error().throw_exception();
    }
  } else {
    throw_exception(eInvalidArgument);
  }
}

Buffer::Buffer(Device device, size_t size, void* native, uint64_t flags)
    : Buffer(device, size, std::shared_ptr<void>(native, [](void*) {}), flags) {}

Buffer::Buffer(Device device, size_t size, std::shared_ptr<void> native, uint64_t flags) {
  if (auto p = GetPlatformImpl(device)) {
    impl_ = p->CreateBuffer(device);
    if (auto r = impl_->Init(size, std::move(native), flags); r.has_error()) {
      impl_.reset();
      r.error().throw_exception();
    }
  } else {
    throw_exception(eInvalidArgument);
  }
}

Device Buffer::GetDevice() const { return impl_ ? impl_->GetDevice() : Device{}; }

Allocator Buffer::GetAllocator() const { return impl_ ? impl_->GetAllocator() : Allocator{}; }

void* Buffer::GetNative(ErrorCode* ec) const {
  return impl_ ? impl_->GetNative(ec) : SetError(ec, eInvalidArgument, nullptr);
}

size_t Buffer::GetSize(ErrorCode* ec) const {
  return impl_ ? impl_->GetSize(ec) : SetError(ec, eInvalidArgument, 0);
}

Buffer::Buffer(Buffer& buffer, size_t offset, size_t size, uint64_t flags) {
  auto impl = buffer.impl_->SubBuffer(offset, size, flags);
  if (!impl) {
    impl.error().throw_exception();
  }
  impl_ = std::move(impl).value();
}

#if 0
int Copy(const void* host_ptr, Buffer& dst, size_t size, size_t dst_offset) {
 Stream stream;
 GetDefaultStream(dst.GetDevice(), &stream);
 if (!stream) {
   return Status(eFail);
 }
 return stream.Copy(host_ptr, dst, size, dst_offset);
}
int Copy(const Buffer& src, void* host_ptr, size_t size, size_t src_offset) {
 Stream stream;
 GetDefaultStream(src.GetDevice(), &stream);
 if (!stream) {
   return Status(eFail);
 }
 return stream.Copy(src, host_ptr, size, src_offset);
}
int Copy(const Buffer& src, Buffer& dst, size_t size, size_t src_offset,
        size_t dst_offset) {
 Stream stream;
 GetDefaultStream(src.GetDevice(), &stream);
 if (!stream) {
   return Status(eFail);
 }
 return stream.Copy(src, dst, size, src_offset, dst_offset);
}
#endif

//////////////////////////////////////////////////
/// Stream

Stream::Stream(Device device, uint64_t flags) {
  if (auto p = GetPlatformImpl(device)) {
    auto impl = p->CreateStream(device);
    if (auto r = impl->Init(flags)) {
      impl_ = std::move(impl);
    } else {
      r.error().throw_exception();
    }
  } else {
    MMDEPLOY_ERROR("{}, {}", device.device_id(), device.platform_id());
    throw_exception(eInvalidArgument);
  }
}

Stream::Stream(Device device, void* native, uint64_t flags)
    : Stream(device, std::shared_ptr<void>(native, [](void*) {}), flags) {}

Stream::Stream(Device device, std::shared_ptr<void> native, uint64_t flags) {
  if (auto p = GetPlatformImpl(device)) {
    auto impl = p->CreateStream(device);
    if (auto r = impl->Init(std::move(native), flags)) {
      impl_ = std::move(impl);
    } else {
      r.error().throw_exception();
    }
  } else {
    throw_exception(eInvalidArgument);
  }
}

Result<void> Stream::Query() {
  if (impl_) {
    return impl_->Query();
  }
  return Status(eInvalidArgument);
}

Result<void> Stream::Wait() {
  if (impl_) {
    return impl_->Wait();
  }
  return Status(eInvalidArgument);
}

Result<void> Stream::DependsOn(Event& event) {
  return impl_ ? impl_->DependsOn(event) : Status(eInvalidArgument);
}

void* Stream::GetNative(ErrorCode* ec) {
  return impl_ ? impl_->GetNative(ec) : SetError(ec, eInvalidArgument, nullptr);
}

Result<void> Stream::Submit(Kernel& kernel) {
  return impl_ ? impl_->Submit(kernel) : Status(eInvalidArgument);
}

Result<void> Stream::Copy(const Buffer& src, Buffer& dst, size_t size, size_t src_offset,
                          size_t dst_offset) {
  if (!impl_) {
    return Status(eInvalidArgument);
  }
  if (size == static_cast<size_t>(-1)) {
    size = src.GetSize();
  }
  if (auto p = GetPlatformImpl(GetDevice())) {
    return p->Copy(src, dst, size, src_offset, dst_offset, *this);
  }
  return Status(eInvalidArgument);
}

Result<void> Stream::Copy(const void* host_ptr, Buffer& dst, size_t size, size_t dst_offset) {
  if (!impl_) {
    return Status(eInvalidArgument);
  }
  if (size == static_cast<size_t>(-1)) {
    size = dst.GetSize();
  }
  auto device = GetDevice();
  if (auto p = GetPlatformImpl(device)) {
    return p->Copy(host_ptr, dst, size, dst_offset, *this);
  }
  return Status(eInvalidArgument);
}

Result<void> Stream::Copy(const Buffer& src, void* host_ptr, size_t size, size_t src_offset) {
  if (!impl_) {
    return Status(eInvalidArgument);
  }
  if (size == static_cast<size_t>(-1)) {
    size = src.GetSize();
  }
  if (auto p = GetPlatformImpl(GetDevice())) {
    return p->Copy(src, host_ptr, size, src_offset, *this);
  }
  return Status(eInvalidArgument);
}

Result<void> Stream::Fill(const Buffer& dst, void* pattern, size_t pattern_size, size_t size,
                          size_t offset) {
  if (!impl_) {
    return Status(eInvalidArgument);
  }
  return Status(eNotSupported);
}

Device Stream::GetDevice() const { return impl_ ? impl_->GetDevice() : Device{}; }

Stream Stream::GetDefault(Device device) {
  Platform platform(device.platform_id());
  assert(platform);
  Stream stream = Access::get<PlatformImpl>(platform).GetDefaultStream(device.device_id()).value();
  return stream;
}

/////////////////////////////////////////////////
/// Event

Event::Event(Device device, uint64_t flags) {
  if (auto p = GetPlatformImpl(device)) {
    auto impl = p->CreateEvent(device);
    if (auto r = impl->Init(flags)) {
      impl_ = std::move(impl);
    } else {
      r.error().throw_exception();
    }
  } else {
    throw_exception(eInvalidArgument);
  }
}

Event::Event(Device device, void* native, uint64_t flags)
    : Event(device, std::shared_ptr<void>(native, [](void*) {}), flags) {}

Event::Event(Device device, std::shared_ptr<void> native, uint64_t flags) {
  if (auto p = GetPlatformImpl(device)) {
    auto impl = p->CreateEvent(device);
    if (auto r = impl->Init(std::move(native), flags)) {
      impl_ = std::move(impl);
    } else {
      r.error().throw_exception();
    }
  } else {
    throw_exception(eInvalidArgument);
  }
}

Result<void> Event::Query() { return impl_ ? impl_->Query() : Status(eInvalidArgument); }

Result<void> Event::Wait() { return impl_ ? impl_->Wait() : Status(eInvalidArgument); }

Result<void> Event::Record(Stream& stream) {
  return impl_ ? impl_->Record(stream) : Status(eInvalidArgument);
}

void* Event::GetNative(ErrorCode* ec) {
  return impl_ ? impl_->GetNative(ec) : SetError(ec, eInvalidArgument, nullptr);
}

Device Event::GetDevice() { return impl_ ? impl_->GetDevice() : Device{}; }

/////////////////////////////////////////////////
/// Kernel

Device Kernel::GetDevice() const { return impl_ ? impl_->GetDevice() : Device{}; }

void* Kernel::GetNative(ErrorCode* ec) {
  return impl_ ? impl_->GetNative(ec) : SetError(ec, eInvalidArgument, nullptr);
}

/////////////////////////////////////////////////
/// PlatformRegistry

int PlatformRegistry::Register(Creator creator) {
  Platform platform(creator());
  auto proposed_id = platform.GetPlatformId();
  std::string name = platform.GetPlatformName();
  if (proposed_id == -1) {
    proposed_id = GetNextId();
    platform.impl_->SetPlatformId(proposed_id);
  } else if (!IsAvailable(proposed_id)) {
    return -1;
  }
  entries_.push_back({name, proposed_id, platform});
  return 0;
}

int PlatformRegistry::AddAlias(const char* name, const char* target) {
  aliases_.emplace_back(name, target);
  return 0;
}

int PlatformRegistry::GetNextId() {
  for (int i = 1;; ++i) {
    if (IsAvailable(i)) {
      return i;
    }
  }
}

bool PlatformRegistry::IsAvailable(int id) {
  for (const auto& entry : entries_) {
    if (entry.id == id) {
      return false;
    }
  }
  return true;
}

int PlatformRegistry::GetPlatform(const char* name, Platform* platform) {
  for (const auto& alias : aliases_) {
    if (name == alias.first) {
      name = alias.second.c_str();
      break;
    }
  }
  for (const auto& entry : entries_) {
    if (entry.name == name) {
      *platform = entry.platform;
      return 0;
    }
  }
  return -1;
}

int PlatformRegistry::GetPlatform(int id, Platform* platform) {
  for (const auto& entry : entries_) {
    if (entry.id == id) {
      *platform = entry.platform;
      return 0;
    }
  }
  return -1;
}

int PlatformRegistry::GetPlatformId(const char* name) {
  for (const auto& alias : aliases_) {
    if (name == alias.first) {
      name = alias.second.c_str();
      break;
    }
  }
  for (const auto& entry : entries_) {
    if (entry.name == name) {
      return entry.id;
    }
  }
  return -1;
}

PlatformImpl* PlatformRegistry::GetPlatformImpl(PlatformId id) {
  for (const auto& entry : entries_) {
    if (entry.id == id) {
      return entry.platform.impl_.get();
    }
  }
  return nullptr;
}

PlatformRegistry& gPlatformRegistry() {
  static PlatformRegistry instance;
  return instance;
}

}  // namespace mmdeploy
