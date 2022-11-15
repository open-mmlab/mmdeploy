// Copyright (c) OpenMMLab. All rights reserved.

#include "jpeg_decoder.h"

#include "common_internal.h"
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codecs/nvjpeg/jpeg_decoder.h"
#include "mmdeploy/core/mpl/structure.h"

using namespace mmdeploy::codecs;
using namespace std;

namespace {
using ResultType = mmdeploy::Structure<mmdeploy_mat_t, mmdeploy::framework::Buffer>;
}

int mmdeploy_jpeg_decoder_create(int device_id, mmdeploy_jpeg_decoder_t* decoder) {
  try {
    auto _decoder = std::make_unique<JPEGDecoder>(device_id);
    *decoder = Cast(_decoder.release());
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

void mmdeploy_jpeg_decoder_destroy(mmdeploy_jpeg_decoder_t decoder) {
  if (decoder != nullptr) {
    delete Cast(decoder);
  }
}

int mmdeploy_jpeg_decoder_apply(mmdeploy_jpeg_decoder_t decoder, const char** raw_data, int* length,
                                int count, mmdeploy_pixel_format_t format,
                                mmdeploy_mat_t** results) {
  if (format != MMDEPLOY_PIXEL_FORMAT_BGR && format != MMDEPLOY_PIXEL_FORMAT_RGB) {
    MMDEPLOY_ERROR(
        "pixel format only support MMDEPLOY_PIXEL_FORMAT_BGR or MMDEPLOY_PIXEL_FORMAT_RGB");
    return MMDEPLOY_E_INVALID_ARG;
  }

  try {
    auto handle = Cast(decoder);
    std::vector<const char*> _raw_data;
    std::vector<int> _length;
    for (int i = 0; i < count; i++) {
      _raw_data.push_back(raw_data[i]);
      _length.push_back(length[i]);
    }
    PixelFormat _format =
        (format == MMDEPLOY_PIXEL_FORMAT_BGR) ? PixelFormat::kBGR : PixelFormat::kRGB;

    auto value = handle->Apply(_raw_data, _length, _format).value();
    auto output = from_value<std::vector<Mat>>(value);

    ResultType r(count);
    auto [_results, buffers] = r.pointers();

    for (int i = 0; i < count; ++i) {
      auto upscale = output[i];
      auto& res = _results[i];
      res.data = upscale.data<uint8_t>();
      buffers[i] = upscale.buffer();
      res.format = (mmdeploy_pixel_format_t)upscale.pixel_format();
      res.height = upscale.height();
      res.width = upscale.width();
      res.channel = upscale.channel();
      res.type = (mmdeploy_data_type_t)upscale.type();
    }

    *results = _results;
    r.release();

    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  MMDEPLOY_E_FAIL;
}

void mmdeploy_jpeg_decoder_release_result(mmdeploy_mat_t* results, int count) {
  ResultType deleter{static_cast<size_t>(count), results};
}
