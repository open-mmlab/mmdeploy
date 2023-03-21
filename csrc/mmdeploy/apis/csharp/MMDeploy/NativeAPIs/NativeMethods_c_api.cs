using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace MMDeploy
{
    /// <summary>
    /// Nativate C methods.
    /// </summary>
    internal static partial class NativeMethods
    {
        #region common.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_context_create(out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_context_create_by_device(string deviceName, int deviceId,
            out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_context_destroy(IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_context_add(IntPtr handle, int type, string name,
            IntPtr obj);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_device_create(string device_name, int device_id,
            out IntPtr device);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_device_destroy(IntPtr device);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_profiler_create(string path, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_profiler_destroy(IntPtr handle);
        #endregion

        #region scheduler.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void* mmdeploy_executor_create_thread();
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void* mmdeploy_executor_create_thread_pool(int num_threads);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_scheduler_destroy(IntPtr handle);
        #endregion

        #region model.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_model_create_by_path(string path, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_model_create(IntPtr buffer, int size, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_model_destroy(IntPtr model);
        #endregion

        #region pose_detector.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_pose_detector_create(IntPtr model, string deviceName,
            int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_pose_detector_create_by_path(string modelPath,
            string deviceName, int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_pose_detector_apply(IntPtr handle, Mat* mats,
            int matCount, CPoseDetect** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_pose_detector_apply_bbox(IntPtr handle, Mat* mats,
            int matCount, Rect* bboxes, int* bbox_count, CPoseDetect** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_pose_detector_release_result(CPoseDetect* results,
            int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_pose_detector_destroy(IntPtr handle);
        #endregion

        #region pose_tracker.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_pose_tracker_create(IntPtr det_model, IntPtr pose_model,
            IntPtr context, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_pose_tracker_destroy(IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_pose_tracker_default_params(IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_pose_tracker_create_state(IntPtr pipeline,
            PoseTracker.Params param, out IntPtr state);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_pose_tracker_destroy_state(IntPtr state);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_pose_tracker_apply(IntPtr handle, IntPtr* state,
            Mat* mats, int* useDet, int count, CPoseTrack** results, int** resultCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_pose_tracker_release_result(CPoseTrack* results,
            int* resultCount, int count);
        #endregion

        #region classifier.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_classifier_create(IntPtr model, string deviceName,
            int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_classifier_create_by_path(string modelPath,
            string deviceName, int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_classifier_apply(IntPtr handle, Mat* mats,
            int matCount, Label** results, int** resultCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_classifier_release_result(Label* results,
            int* resultCount, int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_classifier_destroy(IntPtr handle);
        #endregion

        #region rotated_detector.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_rotated_detector_create(IntPtr model,
            string deviceName, int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_rotated_detector_create_by_path(string modelPath,
            string deviceName, int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_rotated_detector_apply(IntPtr handle, Mat* mats,
            int matCount, RDetect** results, int** resultCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_rotated_detector_release_result(RDetect* results,
            int* resultCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_rotated_detector_destroy(IntPtr handle);
        #endregion

        #region detector.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_detector_create(IntPtr model, string deviceName,
            int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_detector_create_by_path(string modelPath,
            string deviceName, int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_detector_apply(IntPtr handle, Mat* mats,
            int matCount, CDetect** results, int** resultCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_detector_release_result(CDetect* results,
            int* resultCount, int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_detector_destroy(IntPtr handle);
        #endregion

        #region restorer.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_restorer_create(IntPtr model, string deviceName,
            int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_restorer_create_by_path(string modelPath,
            string deviceName, int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_restorer_apply(IntPtr handle, Mat* images,
            int count, Mat** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_restorer_release_result(Mat* results,
            int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_restorer_destroy(IntPtr handle);
        #endregion

        #region segmentor.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_segmentor_create(IntPtr model, string deviceName,
            int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_segmentor_create_by_path(string modelPath,
            string deviceName, int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_segmentor_apply(IntPtr handle, Mat* mats,
            int matCount, CSegment** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_segmentor_release_result(CSegment* results,
            int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_segmentor_destroy(IntPtr handle);
        #endregion

        #region text_detector.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_text_detector_create(IntPtr model, string deviceName,
            int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_text_detector_create_by_path(string modelPath,
            string deviceName, int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_text_detector_apply(IntPtr handle, Mat* mats,
            int matCount, TextDetect** results, int** resultCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_text_detector_release_result(TextDetect* results,
            int* resultCount, int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_text_detector_destroy(IntPtr handle);
        #endregion

        #region text_recognizer.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_text_recognizer_create(IntPtr model, string deviceName,
            int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int mmdeploy_text_recognizer_create_by_path(string modelPath,
            string deviceName, int deviceId, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_text_recognizer_apply(IntPtr handle, Mat* images,
            int count, CTextRecognize** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_text_recognizer_apply_bbox(IntPtr handle,
            Mat* images, int image_count, TextDetect* bboxes, int* bbox_count,
            CTextRecognize** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_text_recognizer_release_result(
            CTextRecognize* results, int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_text_recognizer_destroy(IntPtr handle);
        #endregion
    }
}
