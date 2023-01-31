// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MEDIAIO_H
#define MMDEPLOY_MEDIAIO_H

#include <fstream>
#include <set>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/videoio/videoio.hpp"

namespace utils {
namespace mediaio {

enum class MediaType { kUnknown, kImage, kVideo, kImageList, kWebcam, kFmtStr, kShow };

namespace detail {

static std::string get_extension(const std::string& path) {
  std::string ext;
  for (auto i = (int)path.size() - 1; i >= 0; --i) {
    if (path[i] == '.') {
      ext.push_back(path[i]);
      for (++i; i < path.size(); ++i) {
        ext.push_back((char)std::tolower((unsigned char)path[i]));
      }
      return ext;
    }
  }
  return {};
}

static bool is_video(const std::string& ext) {
  static const std::set<std::string> es{".mp4", ".avi", ".mkv", ".webm", ".mov", ".mpg", ".wmv"};
  return es.count(ext);
}

static bool is_list(const std::string& ext) {
  static const std::set<std::string> es{".txt"};
  return es.count(ext);
}

static bool is_image(const std::string& ext) {
  static const std::set<std::string> es{".jpg", ".jpeg", ".png", ".tif", ".tiff",
                                        ".bmp", ".ppm",  ".pgm", ".webp"};
  return es.count(ext);
}

static bool is_fmtstr(const std::string& str) {
  for (const auto& c : str) {
    if (c == '%') {
      return true;
    }
  }
  return false;
}

}  // namespace detail

class Input;

class InputIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using reference = cv::Mat&;
  using value_type = reference;
  using pointer = void;

 public:
  InputIterator() = default;
  explicit InputIterator(Input& input) : input_(&input) { next(); }
  InputIterator& operator++() {
    next();
    return *this;
  }
  reference operator*() { return frame_; }
  friend bool operator==(const InputIterator& a, const InputIterator& b) {
    return &a == &b || a.is_end() == b.is_end();
  }
  friend bool operator!=(const InputIterator& a, const InputIterator& b) { return !(a == b); }

 private:
  void next();
  bool is_end() const noexcept { return frame_.data != nullptr; }

 private:
  cv::Mat frame_;
  Input* input_{};
};

class Input {
 public:
  explicit Input(const std::string& path, MediaType type = MediaType::kUnknown)
      : path_(path), type_(type) {
    if (type_ == MediaType::kUnknown) {
      auto ext = detail::get_extension(path);
      if (detail::is_image(ext)) {
        type_ = MediaType::kImage;
      } else if (detail::is_video(ext)) {
        type_ = MediaType::kVideo;
      } else if (path.size() == 1 && std::isdigit((unsigned char)path[0])) {
        type_ = MediaType::kWebcam;
      } else if (detail::is_list(ext) || try_image_list(path)) {
        type_ = MediaType::kImageList;
      } else if (try_image(path)) {
        type_ = MediaType::kImage;
      } else if (try_video(path)) {
        type_ = MediaType::kVideo;
      } else {
        std::cout << "unknown file type: " << path << "\n";
      }
    }
    if (type_ != MediaType::kUnknown) {
      if (type_ == MediaType::kVideo) {
        cap_.open(path_);
        if (!cap_.isOpened()) {
          std::cerr << "failed to open video file: " << path_ << "\n";
        }
      } else if (type_ == MediaType::kWebcam) {
        cap_.open(std::stoi(path_));
        if (!cap_.isOpened()) {
          std::cerr << "failed to open camera index: " << path_ << "\n";
        }
        type_ = MediaType::kVideo;
      } else if (type_ == MediaType::kImage) {
        items_ = {path_};
        type_ = MediaType::kImageList;
      } else if (type_ == MediaType::kImageList) {
        if (items_.empty()) {
          items_ = load_image_list(path);
        }
      }
    }
  }
  InputIterator begin() { return InputIterator(*this); }
  InputIterator end() { return {}; }  // NOLINT

  cv::Mat read() {
    cv::Mat img;
    if (type_ == MediaType::kVideo) {
      cap_ >> img;
    } else if (type_ == MediaType::kImageList) {
      while (!img.data && index_ < items_.size()) {
        auto path = items_[index_++];
        img = cv::imread(path);
        if (!img.data) {
          std::cerr << "failed to load image: " << path << "\n";
        }
      }
    }
    return img;
  }

 private:
  static bool try_image(const std::string& path) { return cv::imread(path).data; }

  static bool try_video(const std::string& path) { return cv::VideoCapture(path).isOpened(); }

  static std::vector<std::string> load_image_list(const std::string& path, size_t max_bytes = 0) {
    std::ifstream ifs(path);
    ifs.seekg(0, std::ifstream::end);
    auto size = ifs.tellg();
    ifs.seekg(0, std::ifstream::beg);
    if (max_bytes && size > max_bytes) {
      return {};
    }
    auto strip = [](std::string& s) {
      while (!s.empty() && std::isspace((unsigned char)s.back())) {
        s.pop_back();
      }
    };
    std::vector<std::string> ret;
    std::string line;
    while (std::getline(ifs, line)) {
      strip(line);
      if (!line.empty()) {
        ret.push_back(std::move(line));
      }
    }
    return ret;
  }

  bool try_image_list(const std::string& path) {
    auto items = load_image_list(path, 1 << 20);
    size_t count = 0;
    for (const auto& item : items) {
      if (detail::is_image(detail::get_extension(item)) && ++count > items.size() / 10) {
        items_ = std::move(items);
        return true;
      }
    }
    return false;
  }

 private:
  MediaType type_{MediaType::kUnknown};
  std::string path_;
  std::vector<std::string> items_;
  cv::VideoCapture cap_;
  size_t index_{};
};

inline void InputIterator::next() {
  assert(input_);
  frame_ = input_->read();
}

class Output;

class OutputIterator {
 public:
  using iterator_category = std::output_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using reference = void;
  using value_type = void;
  using pointer = void;

 public:
  explicit OutputIterator(Output& output) : output_(&output) {}

  OutputIterator& operator=(const cv::Mat& frame);

  OutputIterator& operator*() { return *this; }
  OutputIterator& operator++() { return *this; }
  OutputIterator& operator++(int) { return *this; }  // NOLINT

 private:
  Output* output_{};
};

class Output {
 public:
  explicit Output(const std::string& path, int delay, MediaType type = MediaType::kUnknown)
      : path_(path), type_(type), delay_(delay) {
    if (type_ == MediaType::kUnknown) {
      auto ext = detail::get_extension(path);
      if (path_.empty()) {
        type_ = MediaType::kShow;
      } else if (detail::is_image(ext)) {
        if (detail::is_fmtstr(path)) {
          type_ = MediaType::kFmtStr;
        } else {
          type_ = MediaType::kImage;
        }
      } else if (detail::is_video(ext)) {
        type_ = MediaType::kVideo;
      } else {
        std::cout << "unknown file type: " << path << "\n";
      }
    }
  }

  bool write(const cv::Mat& frame) {
    bool exit = false;
    switch (type_) {
      case MediaType::kImage:
        cv::imwrite(path_, frame);
        break;
      case MediaType::kFmtStr: {
        char buf[256];
        snprintf(buf, sizeof(buf), path_.c_str(), frame_id_);
        cv::imwrite(buf, frame);
        break;
      }
      case MediaType::kVideo:
        write_video(frame);
        break;
      case MediaType::kShow:
        cv::imshow("", frame);
        exit = cv::waitKey(delay_) == 'q';
        break;
      default:
        std::cout << "unsupported output media type\n";
        assert(0);
    }
    ++frame_id_;
    return !exit;
  }

  OutputIterator inserter() { return OutputIterator{*this}; }

 private:
  void write_video(const cv::Mat& frame) {
    if (!video_.isOpened()) {
      video_.open(path_, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 30, frame.size());
    }
    video_ << frame;
  }

 private:
  std::string path_;
  MediaType type_{MediaType::kUnknown};
  int delay_{1};
  size_t frame_id_{0};
  cv::VideoWriter video_;
};

OutputIterator& OutputIterator::operator=(const cv::Mat& frame) {
  assert(output_);
  output_->write(frame);
  return *this;
}

}  // namespace mediaio
}  // namespace utils
#endif  // MMDEPLOY_MEDIAIO_H
