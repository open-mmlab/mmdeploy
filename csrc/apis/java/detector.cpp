// Copyright (c) OpenMMLab. All rights reserved.

#include "detector.h"

#include "common.h"

namespace mmdeploy {

class JavaDetector {
 public:
  const char *java_package_name;
  JavaDetector(const char *model_path, const char *device_name, int device_id, const char *package_name) {
    auto status = mmdeploy_detector_create_by_path(model_path, device_name, device_id, &handle_);
    java_package_name = package_name;
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to create detector");
    }
  }
  static int draw_fps(int w, int h, cv::Mat& rgb)
  {
      // resolve moving average
      float avg_fps = 0.f;
      {
          static double t0 = 0.f;
          static float fps_history[10] = {0.f};

          double t1 = ncnn::get_current_time();
          if (t0 == 0.f)
          {
              t0 = t1;
              return 0;
          }

          float fps = 1000.f / (t1 - t0);
          t0 = t1;

          for (int i = 9; i >= 1; i--)
          {
              fps_history[i] = fps_history[i - 1];
          }
          fps_history[0] = fps;

          if (fps_history[9] == 0.f)
          {
              return 0;
          }

          for (int i = 0; i < 10; i++)
          {
              avg_fps += fps_history[i];
          }
          avg_fps /= 10.f;
      }

      char text[32];
      sprintf(text, "%dx%d FPS=%.2f", w, h, avg_fps);

      int baseLine = 0;
      cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

      int y = 0;
      int x = rgb.cols - label_size.width;

      cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

      cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

      return 0;
  }

  extern "C" {
    JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved){
      __android_log_print(ANDROID_LOG_DEBUG,"JNITag","enter jni_onload");
      JNIEnv* env = NULL;
      jint result = -1;
      // ATTENTION JNI_VERSION VALUE!
      if((*vm)->GetEnv(vm,(void**)&env,JNI_VERSION_1_6)!= JNI_OK){
        return result;
      }
      // 定义函数映射关系（参数1：java native函数，参数2：函数描述符，参数3：C函数）
      const JNINativeMethod method[]={
              {"detectDraw","(II[I)Z",(void*)detectDraw},
              {"loadModel","(Ljava/lang/Object;Ljava/lang/StringI)Z",(void*)loadModel},
      };
      jclass jClassName=(*env)->FindClass(env, java_package_name);
      jint ret = (*env)->RegisterNatives(env,jClassName,method, 2);
      if (ret != JNI_OK) {
          __android_log_print(ANDROID_LOG_DEBUG, "JNITag", "jni_register Error");
          return -1;
      }
      return JNI_VERSION_1_6;
    }
    JNIEXPORT jboolean JNICALL detectDraw(JNIEnv* env, jobject thiz, jint jw, jint jh, jintArray jPixArr)
    {
      jint *cPixArr = env->GetIntArrayElements(jPixArr, JNI_FALSE);
      if (cPixArr == NULL) {
          return JNI_FALSE;
      }
      cv::Mat mat_image_src(jh, jw, CV_8UC4, (unsigned char *) cPixArr);
      cv::Mat rgb;
      cvtColor(mat_image_src, rgb, cv::COLOR_RGBA2RGB, 3);

      {
        ncnn::MutexLockGuard g(lock);
        mm_detect_t *bboxes{};
        int *res_count{};
        int status{};
        mm_mat_t mat{rgb.data, rgb.rows, rgb.cols, 3, MM_BGR, MM_INT8};
        status = mmdeploy_detector_apply(detector, &mat, 1, &bboxes, &res_count);
        if (status != MM_SUCCESS) {
            fprintf(stderr, "failed to apply detector, code: %d\n", (int)status);
            return JNI_FALSE;
        }
        static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };
        static const unsigned char colors[19][3] = {
            { 54,  67, 244},
            { 99,  30, 233},
            {176,  39, 156},
            {183,  58, 103},
            {181,  81,  63},
            {243, 150,  33},
            {244, 169,   3},
            {212, 188,   0},
            {136, 150,   0},
            { 80, 175,  76},
            { 74, 195, 139},
            { 57, 220, 205},
            { 59, 235, 255},
            {  7, 193, 255},
            {  0, 152, 255},
            { 34,  87, 255},
            { 72,  85, 121},
            {158, 158, 158},
            {139, 125,  96}
        };

        int color_index = 0;
        for (int i = 0; i < *res_count; i++)
        {
            const mm_detect_t& det_result = bboxes[i];
            // skip detections with invalid bbox size (bbox height or width < 1)
            if ((det_result.bbox.right - det_result.bbox.left) < 1 || (det_result.bbox.bottom - det_result.bbox.top) < 1) {
                continue;
            }
            // skip detections less than specified score threshold
            if (det_result.score < 0.3) {
                continue;
            }
            const unsigned char* color = colors[color_index % 19];
            color_index++;

            cv::Scalar cc(color[0], color[1], color[2]);
            cv::rectangle(rgb, cv::Point{(int)det_result.bbox.left, (int)det_result.bbox.top},
                cv::Point{(int)det_result.bbox.right, (int)det_result.bbox.bottom}, cc, 2);

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[det_result.label_id], det_result.score * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = (int)det_result.bbox.left;
            int y = (int)det_result.bbox.top - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > rgb.cols)
                x = rgb.cols - label_size.width;

            cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

            cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

            cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
        }
        mmdeploy_detector_release_result(bboxes, res_count, 1);
      }
      draw_fps(jw, jh, rgb);
      cvtColor(rgb, mat_image_src, cv::COLOR_RGB2RGBA, 4);
      env->ReleaseIntArrayElements(jPixArr, cPixArr, 0);
      return JNI_TRUE;
    }
    JNIEXPORT jboolean JNICALL loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jstring work_dir, jint modelid)
    {
      AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
      __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);
      __android_log_print(ANDROID_LOG_ERROR, "ncnn", "work_dir jni %s", env->GetStringUTFChars(work_dir,0));
      fprintf(stderr, "work_dir jni %s\n", env->GetStringUTFChars(work_dir,0));
      const char* modeltypes[] =
      {
              "mobilessd",
              "yolo",
              "yolox"
      };
      int status{};
      ncnn::MutexLockGuard g(lock);
      char model_path[256];
      sprintf(model_path, "%s/%s", env->GetStringUTFChars(work_dir,0), modeltypes[(int)modelid]);
      __android_log_print(ANDROID_LOG_ERROR, "ncnn", "work_dir jni model folder: %s", model_path);
      status = mmdeploy_detector_create_by_path(model_path, "cpu", 0, &detector);
      if (status != MM_SUCCESS) {
          __android_log_print(ANDROID_LOG_ERROR, "ncnn", "failed to create detector, code %d, %s", (int)status, getcwd(NULL, 0));
          // fprintf(stderr, "failed to create detector, code: %d\n", (int)status);
          return JNI_FALSE;
      }
      return JNI_TRUE;
    }
  }

  ~JavaDetector() {
    mmdeploy_detector_destroy(handle_);
    handle_ = {};
  }

  private:
    mm_handle_t handle_{};
};
}  // namespace mmdeploy
