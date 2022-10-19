#include "ssd_head.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/model.h"
#include <sstream>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64

namespace mmdeploy::mmdet {

float MIN_SCORE     = 0.4f;
float NMS_THRESHOLD = 0.45f;

float unexpit(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

float expit(float x) { return (float)(1.0 / (1.0 + expf(-x))); }

void decodeCenterSizeBoxes(float* predictions, std::vector<std::vector<float>>& boxPriors)
{
  for (int i = 0; i < NUM_RESULTS; ++i) {
    float ycenter = predictions[i * 4 + 0] / Y_SCALE * boxPriors[2][i] + boxPriors[0][i];
    float xcenter = predictions[i * 4 + 1] / X_SCALE * boxPriors[3][i] + boxPriors[1][i];
    float h       = (float)expf(predictions[i * 4 + 2] / H_SCALE) * boxPriors[2][i];
    float w       = (float)expf(predictions[i * 4 + 3] / W_SCALE) * boxPriors[3][i];

    float ymin = ycenter - h / 2.0f;
    float xmin = xcenter - w / 2.0f;
    float ymax = ycenter + h / 2.0f;
    float xmax = xcenter + w / 2.0f;

    predictions[i * 4 + 0] = ymin;
    predictions[i * 4 + 1] = xmin;
    predictions[i * 4 + 2] = ymax;
    predictions[i * 4 + 3] = xmax;
  }
}

int filterValidResult(float* outputClasses, int (*output)[NUM_RESULTS], int numClasses, float* props)
{
  int   validCount = 0;
  float min_score  = unexpit(MIN_SCORE);

  // debug outputClasses
  for (int i = 0; i < 1; ++i) {
    for (int j = 1; j < numClasses; ++j) {
      float score = outputClasses[i * numClasses + j];
      printf("%.3f, ", score);
    }
    printf("\n");
  }

  // Scale them back to the input size.
  for (int i = 0; i < NUM_RESULTS; ++i) {
    float topClassScore      = (float)(-1000.0);
    int   topClassScoreIndex = -1;

    // Skip the first catch-all class.
    for (int j = 1; j < numClasses; ++j) {
      // x and expit(x) has same monotonicity
      // so compare x and comare expit(x) is same
      // float score = expit(outputClasses[i*numClasses+j]);
      float score = outputClasses[i * numClasses + j];

      if (score > topClassScore) {
        topClassScoreIndex = j;
        topClassScore      = score;
      }
    }

    if (topClassScore >= min_score) {
      output[0][validCount] = i;
      output[1][validCount] = topClassScoreIndex;
      props[validCount]     = expit(outputClasses[i * numClasses + topClassScoreIndex]);
      ++validCount;
    }
  }

  return validCount;
}

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                       float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
  float i = w * h;
  float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
  return u <= 0.f ? 0.f : (i / u);
}


int nms(int validCount, float* outputLocations, int (*output)[NUM_RESULTS])
{
  for (int i = 0; i < validCount; ++i) {
    if (output[0][i] == -1) {
      continue;
    }
    int n = output[0][i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = output[0][j];
      if (m == -1) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 1];
      float ymin0 = outputLocations[n * 4 + 0];
      float xmax0 = outputLocations[n * 4 + 3];
      float ymax0 = outputLocations[n * 4 + 2];

      float xmin1 = outputLocations[m * 4 + 1];
      float ymin1 = outputLocations[m * 4 + 0];
      float xmax1 = outputLocations[m * 4 + 3];
      float ymax1 = outputLocations[m * 4 + 2];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou >= NMS_THRESHOLD) {
        output[0][j] = -1;
      }
    }
  }
  return 0;
}

void sort(int output[][NUM_RESULTS], float* props, int sz)
{
  int i = 0;
  int j = 0;

  if (sz < 2) {
    return;
  }

  for (i = 0; i < sz - 1; i++) {
    int top = i;
    for (j = i + 1; j < sz; j++) {
      if (props[top] < props[j]) {
        top = j;
      }
    }

    if (i != top) {
      int   tmp1     = output[0][i];
      int   tmp2     = output[1][i];
      float prop     = props[i];
      output[0][i]   = output[0][top];
      output[1][i]   = output[1][top];
      props[i]       = props[top];
      output[0][top] = tmp1;
      output[1][top] = tmp2;
      props[top]     = prop;
    }
  }
}

SSDHead::SSDHead(const Value &cfg) : MMDetection(cfg) {
  auto init = [&]() -> Result<void> {
    auto model = cfg["context"]["model"].get<Model>();
    OUTCOME_TRY(auto str_priors, model.ReadFile("box_priors.txt"));
    std::istringstream ss(str_priors);

    priors_.reserve(NUM_SIZE);
    for (int i = 0; i < NUM_SIZE; ++i) {
      std::vector<float> prior(NUM_RESULTS);
      for (int j = 0; j < NUM_RESULTS; ++j) {
        ss >> prior[j];
      }
      priors_.push_back(prior);
    }
    return success();
  };
  init().value();
}

Result<Value> SSDHead::operator()(const Value &prep_res, const Value &infer_res) {
  MMDEPLOY_INFO("prep_res: {}\ninfer_res: {}", prep_res, infer_res);
  try {
    std::vector<Tensor> outputs = GetDetsLabels(prep_res, infer_res);
  } catch (...) {
    return Status(eFail);
  }
  return success();
}

std::vector<Tensor> SSDHead::GetDetsLabels(const Value &prep_res, const Value &infer_res) {
  auto dets = infer_res["dets"].get<Tensor>();
  auto labels = infer_res["labels"].get<Tensor>();
  auto predictions = static_cast<float*>(dets.data());
  auto output_classes = static_cast<float*>(labels.data());
  auto height = prep_res["img_metas"]["img_shape"][1].get<int>();
  auto width = prep_res["img_metas"]["img_shape"][2].get<int>();

  MMDEPLOY_INFO("dets: {}, {}", dets.shape(), dets.data_type());
  MMDEPLOY_INFO("labels: {}, {}", labels.shape(), labels.data_type());

  MMDEPLOY_INFO("before decodeCenterSizeBoxes ...");
  decodeCenterSizeBoxes(predictions, priors_);
  MMDEPLOY_INFO("after decodeCenterSizeBoxes ...");

  int   output[2][NUM_RESULTS];
  float props[NUM_RESULTS];
  memset(output, 0, 2 * NUM_RESULTS);
  memset(props, 0, sizeof(float) * NUM_RESULTS);

  MMDEPLOY_INFO("before filterValidResult ...");
  int validCount = filterValidResult(output_classes, output, NUM_CLASS, props);
  MMDEPLOY_INFO("after filterValidResult ...");

  if (validCount > OBJ_NUMB_MAX_SIZE) {
    MMDEPLOY_ERROR("validCount too much !! {} vs {}", validCount, OBJ_NUMB_MAX_SIZE);
    return {};
  }

  MMDEPLOY_INFO("before sort ...");
  sort(output, props, validCount);
  MMDEPLOY_INFO("after sort ...");

  /* detect nest box */
  MMDEPLOY_INFO("before nms ...");
  nms(validCount, predictions, output);
  MMDEPLOY_INFO("before nms ...");
//  for (int i = 0; i < 100; ++i) {
//    printf("%.3f, %.3f, %.3f, %.3f\n", predictions[i * 4 + 0], predictions[i * 4 + 1], predictions[i * 4 + 2], predictions[i * 4 + 3]);
//  }

  int last_count = 0;
//  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (output[0][i] == -1) {
      continue;
    }
    int n                  = output[0][i];
    int topClassScoreIndex = output[1][i];
    printf("%.3f, %.3f, %.3f, %.3f\n", predictions[n * 4 + 0], predictions[n * 4 + 1], predictions[n * 4 + 2], predictions[n * 4 + 3]);

    int x1 = (int)(predictions[n * 4 + 1] * width);
    int y1 = (int)(predictions[n * 4 + 0] * height);
    int x2 = (int)(predictions[n * 4 + 3] * width);
    int y2 = (int)(predictions[n * 4 + 2] * height);
    // There are a bug show toothbrush always
    if (x1 == 0 && x2 == 0 && y1 == 0 && y2 == 0)
      continue;
//    char* label = labels[topClassScoreIndex];

//    group->results[last_count].box.left   = x1;
//    group->results[last_count].box.top    = y1;
//    group->results[last_count].box.right  = x2;
//    group->results[last_count].box.bottom = y2;
//    group->results[last_count].prop       = props[i];
//    memcpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    printf("ssd result %2d: (%4d, %4d, %4d, %4d), %4.2f\n", i, x1, y1, x2, y2, props[i]/*, label*/);
    last_count++;
  }

//  group->count = last_count;

  return {};
}

REGISTER_CODEBASE_COMPONENT(MMDetection, SSDHead);

} // namespace mmdeploy::mmdet
