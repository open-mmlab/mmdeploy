// Copyright (c) OpenMMLab. All rights reserved.
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

#include "mmdeploy/core/device_impl.h"
#include "mmdeploy/core/types.h"

namespace mmdeploy::framework {

class CpuPlatformImpl : public PlatformImpl {
 public:
  int GetPlatformId() const noexcept override;

  const char* GetPlatformName() const noexcept override;

  shared_ptr<BufferImpl> CreateBuffer(Device device) override;

  shared_ptr<StreamImpl> CreateStream(Device device) override;

  shared_ptr<EventImpl> CreateEvent(Device device) override;

  Result<void> Copy(const void* host_ptr, Buffer dst, size_t size, size_t dst_offset,
                    Stream stream) override;

  Result<void> Copy(Buffer src, void* host_ptr, size_t size, size_t src_offset,
                    Stream stream) override;

  Result<void> Copy(Buffer src, Buffer dst, size_t size, size_t src_offset, size_t dst_offset,
                    Stream stream) override;

  Result<Stream> GetDefaultStream(int32_t device_id) override;

  Device GetDevice(int device_id) const { return Device(GetPlatformId(), device_id); }

 private:
  static bool CheckCopyParam(size_t src_size, size_t dst_size, size_t src_offset, size_t dst_offset,
                             size_t copy_size);

  static Result<void> CopyImpl(const void* src, void* dst, size_t src_size, size_t dst_size,
                               size_t src_offset, size_t dst_offset, size_t size, Stream st);

  Stream default_stream_;
  std::once_flag init_flag_;
};

CpuPlatformImpl& gCpuPlatform();

class CpuHostMemory;

class CpuBufferImpl : public BufferImpl {
 public:
  explicit CpuBufferImpl(Device device);

  Result<void> Init(size_t size, Allocator allocator, size_t alignment, uint64_t flags) override;

  Result<void> Init(size_t size, std::shared_ptr<void> native, uint64_t flags) override;

  Result<BufferImplPtr> SubBuffer(size_t offset, size_t size, uint64_t flags) override;

  void* GetNative(ErrorCode* ec) override;

  Allocator GetAllocator() const override;

  size_t GetSize(ErrorCode* ec) override;

 private:
  std::shared_ptr<CpuHostMemory> memory_;
  size_t offset_{0};
  size_t size_{0};
};

class CpuStreamImpl : public StreamImpl {
 public:
  using Task = std::function<void()>;

  explicit CpuStreamImpl(Device device);

  ~CpuStreamImpl() override;

  Result<void> Init(uint64_t flags) override;

  Result<void> Init(std::shared_ptr<void> native, uint64_t flags) override;

  Result<void> Enqueue(Task task);

  Result<void> DependsOn(Event& event) override;

  Result<void> Query() override;

  Result<void> Wait() override;

  Result<void> Submit(Kernel& kernel) override;

  void* GetNative(ErrorCode* ec) override;

 private:
  void InternalThreadEntry();
  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<Task> task_queue_;
  std::thread thread_;
  Device device_;
  bool abort_{false};
};

class CpuEventImpl : public EventImpl {
 public:
  explicit CpuEventImpl(Device device);

  ~CpuEventImpl() override = default;

  Result<void> Init(uint64_t flags) override;

  Result<void> Init(std::shared_ptr<void> native, uint64_t flags) override;

  Result<void> Query() override;

  Result<void> Record(Stream& stream) override;

  Result<void> Wait() override;

  void* GetNative(ErrorCode* ec) override;

 private:
  void Reset();
  std::shared_future<void> future_;
  std::promise<void> promise_;
};

class CpuKernelImpl : public KernelImpl {
 public:
  using Task = CpuStreamImpl::Task;

  explicit CpuKernelImpl(Device device, Task task) : KernelImpl(device), task_(std::move(task)) {}

  void* GetNative(ErrorCode* ec) override {
    if (ec) *ec = ErrorCode::eSuccess;
    return &task_;
  }

 private:
  Task task_;
};

}  // namespace mmdeploy::framework
