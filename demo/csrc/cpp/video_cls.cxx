
#include <map>
#include <string>

#include "mmdeploy/video_recognizer.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/videoio.hpp"

void SampleFrames(const char* video_path, std::map<int, cv::Mat>& buffer,
                  std::vector<mmdeploy::Mat>& clips, int clip_len, int frame_interval = 1,
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
    clips[i] = img;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  video_cls device_name model_path video_path\n");
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto video_path = argv[3];

  int clip_len = 1;
  int frame_interval = 1;
  int num_clips = 25;

  std::map<int, cv::Mat> buffer;
  std::vector<mmdeploy::Mat> clips;
  mmdeploy::VideoSampleInfo clip_info = {clip_len, num_clips};
  SampleFrames(video_path, buffer, clips, clip_len, frame_interval, num_clips);

  mmdeploy::Model model(model_path);
  mmdeploy::VideoRecognizer recognizer(model, mmdeploy::Device{device_name, 0});

  auto res = recognizer.Apply(clips, clip_info);

  for (const auto& cls : res) {
    fprintf(stderr, "label: %d, score: %.4f\n", cls.label_id, cls.score);
  }

  return 0;
}
