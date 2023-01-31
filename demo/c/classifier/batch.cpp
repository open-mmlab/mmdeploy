#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <string>

#include "mmdeploy/classifier.h"

static int batch_inference(mmdeploy_classifier_t classifier,
                           const std::vector<int>& image_ids,
                           const std::vector<mmdeploy_mat_t>& mats);

int main(int argc, char* argv[]) {
  if (argc < 5) {
    fprintf(stderr, "usage:\n  image_classification device_name dump_model_directory "
            "imagelist.txt batch_size\n");
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];

  mmdeploy_classifier_t classifier{};
  int status{};
  status = mmdeploy_classifier_create_by_path(model_path, device_name, 0, &classifier);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create classifier, code: %d\n", (int)status);
    return 1;
  }

  // `file_path` is the path of an image list file
  std::string file_path = argv[3];
  const int batch = std::stoi(argv[argc-1]);

  // read image paths from the file
  std::ifstream ifs(file_path);
  std::string img_path;
  std::vector<std::string> img_paths;
  while (ifs >> img_path) {
    img_paths.emplace_back(std::move(img_path));
  }

  // read images and process batch inference
  std::vector<cv::Mat> images;
  std::vector<int> image_ids;
  std::vector<mmdeploy_mat_t> mats;
  for (int i = 0; i < (int)img_paths.size(); ++i) {
    auto img = cv::imread(img_paths[i]);
    if (!img.data) {
      fprintf(stderr, "failed to load image: %s\n", img_paths[i].c_str());
      continue;
    }
    images.push_back(img);
    image_ids.push_back(i);
    mmdeploy_mat_t mat{
        img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};
    mats.push_back(mat);

    // process batch inference
    if ((int)mats.size() == batch) {
      if (batch_inference(classifier, image_ids, mats) != 0) {
        continue;
      }
      // clear buffer for next batch
      mats.clear();
      image_ids.clear();
      images.clear();
    }
  }
  // process batch inference if there are still unhandled images
  if (!mats.empty()) {
    (void)batch_inference(classifier, image_ids, mats);
  }

  mmdeploy_classifier_destroy(classifier);

  return 0;
}


int batch_inference(mmdeploy_classifier_t classifier, const std::vector<int>& image_ids,
                    const std::vector<mmdeploy_mat_t>& mats) {
  mmdeploy_classification_t* res{};
  int* res_count{};
  auto status = mmdeploy_classifier_apply(classifier, mats.data(), (int)mats.size(),
                                          &res, &res_count);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply classifier to batch images %d, code: %d\n",
            (int)mats.size(), (int)status);
    return 1;
  }
  // print the inference results
  auto res_ptr = res;
  for (int j = 0; j < (int)mats.size(); ++j) {
    fprintf(stderr, "results in the %d-th image:\n", image_ids[j]);
    for (int k = 0; k < res_count[j]; ++k, ++res_ptr) {
      fprintf(stderr, "  label: %d, score: %.4f\n", res_ptr->label_id, res_ptr->score);
    }
  }
  // release results buffer
  mmdeploy_classifier_release_result(res, res_count, (int)mats.size());
  return 0;
}
