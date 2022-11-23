// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_VIDEO_RECOGNIZER_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_VIDEO_RECOGNIZER_HPP_

#include "mmdeploy/common.hpp"
#include "mmdeploy/video_recognizer.h"

namespace mmdeploy {

namespace cxx {

using VideoRecognition = mmdeploy_video_recognition_t;
using VideoSampleInfo = mmdeploy_video_sample_info_t;

class VideoRecognizer : public NonMovable {
 public:
  VideoRecognizer(const Model& model, const Context& context) {
    auto ec = mmdeploy_video_recognizer_create_v2(model, context, &recognizer_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  ~VideoRecognizer() {
    if (recognizer_) {
      mmdeploy_video_recognizer_destroy(recognizer_);
      recognizer_ = {};
    }
  }

  using Result = Result_<VideoRecognition>;

  std::vector<Result> Apply(Span<const std::vector<Mat>> videos,
                            Span<const VideoSampleInfo> infos) {
    if (videos.empty()) {
      return {};
    }

    int video_count = videos.size();

    VideoRecognition* results{};
    int* result_count{};
    std::vector<Mat> images;
    std::vector<VideoSampleInfo> video_info;
    for (int i = 0; i < videos.size(); i++) {
      for (auto& mat : videos[i]) {
        images.push_back(mat);
      }
      video_info.push_back(infos[i]);
    }

    auto ec =
        mmdeploy_video_recognizer_apply(recognizer_, reinterpret(images.data()), video_info.data(),
                                        video_count, &results, &result_count);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::vector<Result> rets;
    rets.reserve(video_count);

    std::shared_ptr<VideoRecognition> data(results, [result_count, count = video_count](auto p) {
      mmdeploy_video_recognizer_release_result(p, result_count, count);
    });

    size_t offset = 0;
    for (size_t i = 0; i < video_count; ++i) {
      offset += rets.emplace_back(offset, result_count[i], data).size();
    }

    return rets;
  }

  Result Apply(const std::vector<Mat>& video, const VideoSampleInfo info) {
    return Apply(Span{video}, Span{info})[0];
  }

 private:
  mmdeploy_video_recognizer_t recognizer_{};
};

}  // namespace cxx

using cxx::VideoRecognition;
using cxx::VideoRecognizer;
using cxx::VideoSampleInfo;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_VIDEO_RECOGNIZER_HPP_
