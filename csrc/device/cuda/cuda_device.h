// Copyright (c) OpenMMLab. All rights reserved.

#include <any>
#include <mutex>

#include "core/device_impl.h"
#include "core/types.h"
#include "cuda.h"
#include "cuda_runtime.h"

namespace mmdeploy {

using CudaTask = std::function<void(cudaStream_t)>;

class CudaPlatformImpl : public PlatformImpl {
 public:
  CudaPlatformImpl();

  const char* GetPlatformName() const noexcept override { return "cuda"; }

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

  Allocator GetDefaultAllocator(int32_t device_id);

  Device GetDevice(int device_id) { return Device(platform_id_, device_id); }

 private:
  static bool CheckCopyParam(size_t src_size, size_t dst_size, size_t src_offset, size_t dst_offset,
                             size_t copy_size);

  static bool CheckCopyDevice(const Device& src, const Device& dst, const Device& st);

  static Result<void> CopyImpl(Device device, const void* src, void* dst, size_t src_size,
                               size_t dst_size, size_t src_offset, size_t dst_offset, size_t size,
                               Stream st);

  class PerDeviceData {
   public:
    explicit PerDeviceData(int device_id) : device_id_(device_id) {}
    void init();
    Stream& default_stream() {
      init();
      return default_stream_;
    }
    Allocator& default_allocator() {
      init();
      return default_allocator_;
    }

   private:
    int device_id_;
    std::once_flag init_flag_;
    Stream default_stream_;
    Allocator default_allocator_;
  };

  std::vector<std::unique_ptr<PerDeviceData>> per_device_data_storage_;
  std::vector<PerDeviceData*> per_device_data_;
};

CudaPlatformImpl& gCudaPlatform();

class CudaDeviceMemory;

class CudaBufferImpl : public BufferImpl {
 public:
  explicit CudaBufferImpl(Device device);

  Result<void> Init(size_t size, Allocator allocator, size_t alignment, uint64_t flags) override;

  Result<void> Init(size_t size, std::shared_ptr<void> native, uint64_t flags) override;

  Result<BufferImplPtr> SubBuffer(size_t offset, size_t size, uint64_t flags) override;

  void* GetNative(ErrorCode* ec) override;

  Allocator GetAllocator() const override;

  size_t GetSize(ErrorCode* ec) override;

 private:
  std::shared_ptr<CudaDeviceMemory> memory_;
  size_t offset_{0};
  size_t size_{0};
};

class CudaStreamImpl : public StreamImpl {
 public:
  explicit CudaStreamImpl(Device device);

  ~CudaStreamImpl() override;

  Result<void> Init(uint64_t flags) override;

  Result<void> Init(std::shared_ptr<void> native, uint64_t flags) override;

  Result<void> DependsOn(Event& event) override;

  Result<void> Query() override;

  Result<void> Wait() override;

  Result<void> Submit(Kernel& kernel) override;

  void* GetNative(ErrorCode* ec) override;

 private:
  cudaStream_t stream_;
  bool owned_stream_;
  std::shared_ptr<void> external_;
};

class CudaEventImpl : public EventImpl {
 public:
  explicit CudaEventImpl(Device device);

  ~CudaEventImpl() override;

  Result<void> Init(uint64_t flags) override;

  Result<void> Init(std::shared_ptr<void> native, uint64_t flags) override;

  Result<void> Query() override;

  Result<void> Record(Stream& stream) override;

  Result<void> Wait() override;

  void* GetNative(ErrorCode* ec) override;

 private:
  cudaEvent_t event_;
  bool owned_event_;
  std::shared_ptr<void> external_;
};

class CudaKernelImpl : public KernelImpl {
 public:
  explicit CudaKernelImpl(Device device, CudaTask task);

  void* GetNative(ErrorCode* ec) override;

 private:
  CudaTask task_;
};

}  // namespace mmdeploy
