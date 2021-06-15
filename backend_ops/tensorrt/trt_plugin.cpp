#include "scatternd/trt_scatternd.hpp"

REGISTER_TENSORRT_PLUGIN(ONNXScatterNDDynamicCreator);

extern "C" {
bool initLibMMCVInferPlugins() { return true; }
}  // extern "C"
