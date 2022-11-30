// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codecs/nvdec/frame_queue.h"

#include <chrono>
#include <cstring>
#include <thread>

#include "mmdeploy/core/logger.h"

namespace mmdeploy {

namespace nvdec {

void FrameQueue::Init(int max_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  max_size_ = max_size;
  display_queue_.resize(max_size);
  is_frame_in_use_ = new volatile int[max_size];
  std::memset((void*)is_frame_in_use_, 0, sizeof(*is_frame_in_use_) * max_size);
}

void FrameQueue::Enqueue(const CUVIDPARSERDISPINFO* disp_info) {
  is_frame_in_use_[disp_info->picture_index] = true;

  do {
    bool placed = false;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (n_frame_in_queue_ < max_size_) {
        int write_pos = (read_pos_ + n_frame_in_queue_) % max_size_;
        display_queue_[write_pos] = *disp_info;
        n_frame_in_queue_++;
        placed = true;
      }
    }

    if (placed) {
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (!end_of_decode_);
}

bool FrameQueue::Dequeue(CUVIDPARSERDISPINFO* disp_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (n_frame_in_queue_ > 0) {
    int entry = read_pos_;
    *disp_info = display_queue_.at(entry);
    read_pos_ = (entry + 1) % max_size_;
    n_frame_in_queue_--;
    return true;
  }
  return false;
}

void FrameQueue::ReleaseFrame(const CUVIDPARSERDISPINFO* disp_info) {
  is_frame_in_use_[disp_info->picture_index] = false;
}

bool FrameQueue::IsInUse(int picture_index) { return (is_frame_in_use_[picture_index] != 0); }

bool FrameQueue::WaitUntilFrameAvailable(int picture_index) {
  while (IsInUse(picture_index)) {
    // Decoder is getting too far ahead from display
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    if (end_of_decode_) {
      return false;
    }
  }
  return true;
}

FrameQueue::~FrameQueue() { delete[] is_frame_in_use_; }

}  // namespace nvdec
}  // namespace mmdeploy