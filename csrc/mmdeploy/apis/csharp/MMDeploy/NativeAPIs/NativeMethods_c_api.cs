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
