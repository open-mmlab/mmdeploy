#include <fstream>
#include <map>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <set>
#include <string>
#include <vector>

#include "mmdeploy/video_recognizer.h"

void SampleFrames(const char* video_path, std::map<int, cv::Mat>& buffer,
                  std::vector<mmdeploy_mat_t>& clips, int clip_len, int frame_interval = 1,
                  int num_clips = 1) {
  cv::VideoCapture cap = cv::VideoCapture(video_path);
  if (!cap.isOpened()) {
    fprintf(stderr, "failed to load video: %s\n", video_path);
    exit(1);
  }

  int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  printf("num_frames %d\n", num_frames);

  int ori_clip_len = clip_len * frame_interval;
  float avg_interval = (num_frames - ori_clip_len + 1.f) / num_clips;
  std::vector<int> frame_inds;
  for (int i = 0; i < num_clips; i++) {
    int clip_offset = i * avg_interval + avg_interval / 2.0;
    for (int j = 0; j < clip_len; j++) {
      int ind = (j * frame_interval + clip_offset) % num_frames;
      if (num_frames <= ori_clip_len - 1) {
        ind = j % num_frames;
      }
      frame_inds.push_back(ind);
    }
  }

  std::vector<int> unique_inds(frame_inds.begin(), frame_inds.end());
  std::sort(unique_inds.begin(), unique_inds.end());
  auto last = std::unique(unique_inds.begin(), unique_inds.end());
  unique_inds.erase(last, unique_inds.end());

  int ind = 0;
  for (int i = 0; i < unique_inds.size(); i++) {
    int tid = unique_inds[i];
    cv::Mat frame;
    while (ind < tid) {
      cap.read(frame);
      ind++;
    }
    cap.read(frame);
    buffer[tid] = frame;
    ind++;
  }

  clips.resize(frame_inds.size());
  for (int i = 0; i < frame_inds.size(); i++) {
    auto& img = buffer[frame_inds[i]];
    mmdeploy_mat_t mat{
        img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};
    clips[i] = mat;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 7) {
    fprintf(stderr,
            "usage:\n  video_recognition device_name dump_model_directory video_path clip_len "
            "frame_interval num_clips \n");
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto video_path = argv[3];

  int clip_len = std::stoi(argv[4]);
  int frame_interval = std::stoi(argv[5]);
  int num_clips = std::stoi(argv[6]);

  std::map<int, cv::Mat> buffer;
  std::vector<mmdeploy_mat_t> clips;
  std::vector<mmdeploy_video_sample_info_t> clip_info;
  SampleFrames(video_path, buffer, clips, clip_len, frame_interval, num_clips);
  clip_info.push_back({clip_len, num_clips});

  mmdeploy_video_recognizer_t recognizer{};
  int status{};
  status = mmdeploy_video_recognizer_create_by_path(model_path, device_name, 0, &recognizer);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create recognizer, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_video_recognition_t* res{};
  int* res_count{};
  status = mmdeploy_video_recognizer_apply(recognizer, clips.data(), clip_info.data(), 1, &res,
                                           &res_count);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply classifier, code: %d\n", (int)status);
    return 1;
  }

  for (int i = 0; i < res_count[0]; ++i) {
    fprintf(stderr, "label: %d, score: %.4f\n", res[i].label_id, res[i].score);
  }

  mmdeploy_video_recognizer_release_result(res, res_count, 1);

  mmdeploy_video_recognizer_destroy(recognizer);

  return 0;
}
