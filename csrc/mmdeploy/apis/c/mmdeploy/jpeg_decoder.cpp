// Copyright (c) OpenMMLab. All rights reserved.

#include "jpeg_decoder.h"

#include "common_internal.h"
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codecs/nvjpeg/jpeg_decoder.h"
#include "mmdeploy/core/mpl/structure.h"
#include "mmdeploy/core/utils/device_utils.h"

using namespace mmdeploy;
using namespace std;

namespace {

mmdeploy_jpeg_decoder_t Cast(codecs::JPEGDecoder* decoder) {
  return reinterpret_cast<mmdeploy_jpeg_decoder_t>(decoder);
}

codecs::JPEGDecoder* Cast(mmdeploy_jpeg_decoder_t decoder) {
  return reinterpret_cast<codecs::JPEGDecoder*>(decoder);
}

using ResultType = mmdeploy::Structure<mmdeploy_mat_t, mmdeploy::framework::Buffer>;

}  // namespace

int mmdeploy_jpeg_decoder_create(int device_id, mmdeploy_jpeg_decoder_t* decoder) {
  try {
    auto _decoder = std::make_unique<codecs::JPEGDecoder>(device_id);
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
                                mmdeploy_mat_t** dev_results, mmdeploy_mat_t** host_results) {
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
    auto dev_output = from_value<std::vector<Mat>>(value);
    std::vector<Mat> host_output;

    if (host_results) {
      host_output.reserve(dev_output.size());
      auto stream = Stream::GetDefault(dev_output[0].device());
      auto device = Device("cpu");
      for (int i = 0; i < dev_output.size(); i++) {
        auto host_mat = MakeAvailableOnDevice(dev_output[i], device, stream).value();
        host_output.push_back(std::move(host_mat));
      }
      stream.Wait().value();
    }

    auto CopyToResults = [count](ResultType& r, std::vector<Mat>& output,
                                 mmdeploy_mat_t** results) {
      if (results == nullptr) {
        return;
      }

      auto [_results, buffers] = r.pointers();
      for (int i = 0; i < count; ++i) {
        auto mat = output[i];
        auto& res = _results[i];
        res.data = mat.data<uint8_t>();
        buffers[i] = mat.buffer();
        res.format = (mmdeploy_pixel_format_t)mat.pixel_format();
        res.height = mat.height();
        res.width = mat.width();
        res.channel = mat.channel();
        res.type = (mmdeploy_data_type_t)mat.type();
      }

      *results = _results;
    };

    ResultType r_d(count);
    ResultType r_h(count);
    CopyToResults(r_d, dev_output, dev_results);
    CopyToResults(r_h, host_output, host_results);

    if (dev_results) {
      r_d.release();
    }
    if (host_results) {
      r_h.release();
    }

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
