// Copyright (c) OpenMMLab. All rights reserved.

#include "video_decoder.h"

#include "common_internal.h"
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codecs/decoder.h"
#include "mmdeploy/core/mpl/structure.h"
#include "mmdeploy/core/utils/device_utils.h"

using namespace mmdeploy;
using namespace std;

namespace {
mmdeploy_video_decoder_t Cast(VideoDecoder* decoder) {
  return reinterpret_cast<mmdeploy_video_decoder_t>(decoder);
}

VideoDecoder* Cast(mmdeploy_video_decoder_t decoder) {
  return reinterpret_cast<VideoDecoder*>(decoder);
}

using ResultType = mmdeploy::Structure<mmdeploy_mat_t, mmdeploy::framework::Buffer>;

void CopyDecoderOutput(framework::Mat& dev, mmdeploy_mat_t** dev_results,
                       mmdeploy_mat_t** host_results) {
  auto stream = Stream::GetDefault(dev.device());
  framework::Mat host;
  if (host_results) {
    auto device = Device("cpu");
    host = MakeAvailableOnDevice(dev, device, stream).value();
    stream.Wait().value();
  }

  auto CopyToResults = [](ResultType& r, framework::Mat& output, mmdeploy_mat_t** results) {
    if (results == nullptr) {
      return;
    }

    auto [_results, buffers] = r.pointers();
    auto mat = output;
    auto& res = _results[0];
    res.data = mat.data<uint8_t>();
    buffers[0] = mat.buffer();
    res.format = (mmdeploy_pixel_format_t)mat.pixel_format();
    res.height = mat.height();
    res.width = mat.width();
    res.channel = mat.channel();
    res.type = (mmdeploy_data_type_t)mat.type();

    *results = _results;
  };

  ResultType r_d(1);
  ResultType r_h(1);
  CopyToResults(r_d, dev, dev_results);
  CopyToResults(r_h, host, host_results);

  if (dev_results) {
    r_d.release();
  }
  if (host_results) {
    r_h.release();
  }
}

}  // namespace

int mmdeploy_video_decoder_create(mmdeploy_video_decoder_params_t params, const char* device_name,
                                  int device_id, mmdeploy_video_decoder_t* decoder) {
  try {
    auto cfg = Value{};
    cfg["path"] = params.path;
    cfg["device_id"] = device_id;
    cfg["pixel_format"] = (int)params.format;
    auto creator = gRegistry<VideoDecoder>().Get(device_name);
    if (!creator) {
      MMDEPLOY_ERROR("Video decoder creator for '{}' not found. Available video decoders: {}",
                     device_name, gRegistry<VideoDecoder>().List());
      return MMDEPLOY_E_INVALID_ARG;
    }
    auto _decoder = creator->Create(cfg);
    *decoder = Cast((VideoDecoder*)_decoder.release());
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

int mmdeploy_video_decoder_info(mmdeploy_video_decoder_t decoder, mmdeploy_video_info_t* info) {
  try {
    auto handle = Cast(decoder);
    VideoInfo _info;
    if (handle->GetInfo(_info)) {
      info->width = _info.width;
      info->height = _info.height;
      info->fourcc = _info.fourcc;
      info->fps = _info.fps;
      return MMDEPLOY_SUCCESS;
    }
    return MMDEPLOY_E_FAIL;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

void mmdeploy_video_decoder_destroy(mmdeploy_video_decoder_t decoder) {
  if (decoder != nullptr) {
    delete Cast(decoder);
  }
}

int mmdeploy_video_decoder_read(mmdeploy_video_decoder_t decoder, mmdeploy_mat_t** dev_results,
                                mmdeploy_mat_t** host_results) {
  try {
    auto handle = Cast(decoder);
    Mat output;
    if (handle->Read(output)) {
      CopyDecoderOutput(output, dev_results, host_results);
      return MMDEPLOY_SUCCESS;
    }
    return MMDEPLOY_E_FAIL;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

int mmdeploy_video_decoder_grab(mmdeploy_video_decoder_t decoder) {
  try {
    auto handle = Cast(decoder);
    if (handle->Grab()) {
      return MMDEPLOY_SUCCESS;
    }
    return MMDEPLOY_E_FAIL;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

int mmdeploy_video_decoder_retrieve(mmdeploy_video_decoder_t decoder, mmdeploy_mat_t** dev_results,
                                    mmdeploy_mat_t** host_results) {
  try {
    auto handle = Cast(decoder);
    Mat output;
    if (handle->Read(output)) {
      CopyDecoderOutput(output, dev_results, host_results);
      return MMDEPLOY_SUCCESS;
    }
    return MMDEPLOY_E_FAIL;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

void mmdeploy_video_decoder_release_result(mmdeploy_mat_t* results, int count) {
  ResultType deleter{static_cast<size_t>(count), results};
}
