// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODECS_NVDEC_FRAME_QUEUE_H_
#define MMDEPLOY_SRC_CODECS_NVDEC_FRAME_QUEUE_H_

#include <nvcuvid.h>

#include <mutex>
#include <vector>

namespace mmdeploy {

namespace nvdec {

class FrameQueue {
 public:
  ~FrameQueue();

  void Init(int max_size);

  void Enqueue(const CUVIDPARSERDISPINFO* disp_info);

  bool Dequeue(CUVIDPARSERDISPINFO* disp_info);

  void EndDecode() { end_of_decode_ = 1; }

  bool IsEnd() { return end_of_decode_; }

  void ReleaseFrame(const CUVIDPARSERDISPINFO* disp_info);

  bool IsInUse(int picture_index);

  bool WaitUntilFrameAvailable(int picture_index);

 private:
  std::mutex mutex_;
  int n_frame_in_queue_;
  int read_pos_;
  int max_size_;
  std::vector<CUVIDPARSERDISPINFO> display_queue_;
  volatile int* is_frame_in_use_{nullptr};
  volatile int end_of_decode_{0};
};

}  // namespace nvdec

}  // namespace mmdeploy

#endif  //  MMDEPLOY_SRC_CODECS_NVDEC_FRAME_QUEUE_H_