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
        public static extern unsafe int mmdeploy_pose_detector_apply(IntPtr handle, MmMat* mats,
            int matCount, CMmPoseDetect** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_pose_detector_apply_bbox(IntPtr handle, MmMat* mats,
            int matCount, MmRect* bboxes, int* bbox_count, CMmPoseDetect** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_pose_detector_release_result(CMmPoseDetect* results,
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
        public static extern unsafe int mmdeploy_classifier_apply(IntPtr handle, MmMat* mats,
            int matCount, MmClass** results, int** resultCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_classifier_release_result(MmClass* results,
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
        public static extern unsafe int mmdeploy_detector_apply(IntPtr handle, MmMat* mats,
            int matCount, CMmDetect** results, int** resultCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_detector_release_result(CMmDetect* results,
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
        public static extern unsafe int mmdeploy_restorer_apply(IntPtr handle, MmMat* images,
            int count, MmMat** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_restorer_release_result(MmMat* results,
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
        public static extern unsafe int mmdeploy_segmentor_apply(IntPtr handle, MmMat* mats,
            int matCount, CMmSegment** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_segmentor_release_result(CMmSegment* results,
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
        public static extern unsafe int mmdeploy_text_detector_apply(IntPtr handle, MmMat* mats,
            int matCount, MmTextDetect** results, int** resultCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_text_detector_release_result(MmTextDetect* results,
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
        public static extern unsafe int mmdeploy_text_recognizer_apply(IntPtr handle, MmMat* images,
            int count, CMmTextRecognize** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int mmdeploy_text_recognizer_apply_bbox(IntPtr handle,
            MmMat* images, int image_count, MmTextDetect* bboxes, int* bbox_count,
            CMmTextRecognize** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void mmdeploy_text_recognizer_release_result(
            CMmTextRecognize* results, int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void mmdeploy_text_recognizer_destroy(IntPtr handle);
        #endregion
    }
}
