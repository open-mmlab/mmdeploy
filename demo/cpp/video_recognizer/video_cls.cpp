
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/videoio.hpp>
#include <string>

#include "mmdeploy/video_recognizer.hpp"

void SampleFrames(const char* video_path, std::map<int, cv::Mat>& buffer,
                  std::vector<mmdeploy::Mat>& clips, int clip_len, int frame_interval = 1,
                  int num_clips = 1) {
  cv::VideoCapture cap = cv::VideoCapture(video_path);
  if (!cap.isOpened()) {
    std::cerr << "failed to load video: " << video_path << std::endl;
    exit(1);
  }

  int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  std::cout << "num_frames: " << num_frames << std::endl;

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
  for (int tid : unique_inds) {
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
    clips[i] = img;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage:" << std::endl
              << "  ./video_cls device_name sdk_model_path "
                 "video_path [--profile]"
              << std::endl;
    return 1;
  }

  auto device_name = argv[1];
  auto model_path = argv[2];
  auto video_path = argv[3];
  auto profile = argc > 4 ? std::string("--profile") == argv[argc - 1] : false;

  mmdeploy::Context context(mmdeploy::Device{device_name, 0});
  mmdeploy::Profiler profiler("profiler.bin");
  if (profile) {
    context.Add(profiler);
  }

  mmdeploy::VideoRecognizer recognizer( mmdeploy::Model{model_path}, context);

  int clip_len = 1;
  int frame_interval = 1;
  int num_clips = 25;

  std::map<int, cv::Mat> buffer;
  std::vector<mmdeploy::Mat> clips;
  mmdeploy::VideoSampleInfo clip_info = {clip_len, num_clips};
  SampleFrames(video_path, buffer, clips, clip_len, frame_interval, num_clips);

  auto res = recognizer.Apply(clips, clip_info);

  for (const auto& cls : res) {
    std::cout << "label: " << cls.label_id << ", score: " << std::fixed << std::setprecision(2)
              << cls.score << std::endl;
  }

  return 0;
}
