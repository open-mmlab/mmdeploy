#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "mmdeploy/detector.h"

static int batch_inference(mmdeploy_detector_t detector, std::vector<cv::Mat>& images,
                           const std::vector<int>& image_ids,
                           const std::vector<mmdeploy_mat_t>& mats);

static void visualize_detection(const std::string& output_name, cv::Mat& image,
                                const mmdeploy_detection_t* bboxes_ptr, int bboxes_num);

int main(int argc, char* argv[]) {
  if (argc < 5) {
    fprintf(stderr, "usage:\n  object_detection device_name sdk_model_path "
            "file_path batch_size\n");
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];

  mmdeploy_detector_t detector{};
  int status{};
  status = mmdeploy_detector_create_by_path(model_path, device_name, 0, &detector);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create detector, code: %d\n", (int)status);
    return 1;
  }

  // file_path is the path of an image list file
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
      if (batch_inference(detector, images, image_ids, mats) != 0) {
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
    (void)batch_inference(detector, images, image_ids, mats);
  }

  mmdeploy_detector_destroy(detector);
  return 0;
}

int batch_inference(mmdeploy_detector_t detector, std::vector<cv::Mat>& images,
                    const std::vector<int>& image_ids,
                    const std::vector<mmdeploy_mat_t>& mats) {
  mmdeploy_detection_t* bboxes{};
  int* res_count{};
  auto status = mmdeploy_detector_apply(detector, mats.data(), mats.size(), &bboxes, &res_count);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply detector, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_detection_t* bboxes_ptr = bboxes;
  for (int i = 0; i < (int)mats.size(); ++i) {
    fprintf(stdout, "results in the %d-th image:\n  bbox_count=%d\n", image_ids[i], res_count[i]);
    const std::string output_name = "output_detection_" + std::to_string(image_ids[i]) + ".png";
    visualize_detection(output_name, images[i], bboxes_ptr, res_count[i]);
    bboxes_ptr = bboxes_ptr + res_count[i];
  }

  mmdeploy_detector_release_result(bboxes, res_count, mats.size());
  return 0;
}


void visualize_detection(const std::string& output_name, cv::Mat& image,
                         const mmdeploy_detection_t* bboxes_ptr, int bbox_num) {
  for (int i = 0; i < bbox_num; ++i, ++bboxes_ptr) {
    const auto& box = bboxes_ptr->bbox;
    const auto& mask = bboxes_ptr->mask;

    fprintf(stdout,
            "  box %d, left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, "
            "label=%d, score=%.4f\n",
            i, box.left, box.top, box.right, box.bottom, bboxes_ptr->label_id, bboxes_ptr->score);

    // skip detections with invalid bbox size (bbox height or width < 1)
    if ((box.right - box.left) < 1 || (box.bottom - box.top) < 1) {
      continue;
    }

    // skip detections less than specified score threshold
    if (bboxes_ptr->score < 0.3) {
      continue;
    }

    // generate mask overlay if model exports masks
    if (mask != nullptr) {
      fprintf(stdout, "mask %d, height=%d, width=%d\n", i, mask->height, mask->width);

      cv::Mat imgMask(mask->height, mask->width, CV_8UC1, &mask->data[0]);
      auto x0 = std::max(std::floor(box.left) - 1, 0.f);
      auto y0 = std::max(std::floor(box.top) - 1, 0.f);
      cv::Rect roi((int)x0, (int)y0, mask->width, mask->height);

      // split the RGB channels, overlay mask to a specific color channel
      cv::Mat ch[3];
      split(image, ch);
      int col = 0;
      cv::bitwise_or(imgMask, ch[col](roi), ch[col](roi));
      merge(ch, 3, image);
    }

    cv::rectangle(image, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }
  cv::imwrite(output_name, image);
}
