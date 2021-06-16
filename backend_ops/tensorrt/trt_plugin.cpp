#include "nms/trt_nms.hpp"
#include "roi_align/trt_roi_align.hpp"
#include "scatternd/trt_scatternd.hpp"

REGISTER_TENSORRT_PLUGIN(NonMaxSuppressionDynamicCreator);
REGISTER_TENSORRT_PLUGIN(ONNXScatterNDDynamicCreator);
REGISTER_TENSORRT_PLUGIN(RoIAlignPluginDynamicCreator);

extern "C" {
bool initLibMMCVInferPlugins() { return true; }
}  // extern "C"
