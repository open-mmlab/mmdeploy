using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace MMDeploySharp
{
    public static partial class NativeMethods
    {
        #region model.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_model_create_by_path(String path, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_model_create(IntPtr buffer, int size, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void c_mmdeploy_model_destroy(IntPtr model);
        #endregion

        #region pose_detector.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_pose_detector_create(IntPtr model, String device_name,
            int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_pose_detector_create_by_path(String model_path,
            String device_name, int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_pose_detector_apply(IntPtr handle, mm_mat_t* mats,
            int mat_count, mm_pose_detect_t** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_pose_detector_apply_bbox(IntPtr handle, mm_mat_t* mats,
            int mat_count, mm_rect_t* bboxes, int* bbox_count, mm_pose_detect_t** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_pose_detector_release_result(mm_pose_detect_t* results,
            int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void c_mmdeploy_pose_detector_destroy(IntPtr handle);
        #endregion

        #region classifier.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_classifier_create(IntPtr model, String device_name,
            int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_classifier_create_by_path(String model_path,
            String device_name, int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_classifier_apply(IntPtr handle, mm_mat_t* mats,
            int mat_count, mm_class_t** results, int** result_count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void c_mmdeploy_classifier_release_result(mm_class_t* results,
            int* result_count, int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void c_mmdeploy_classifier_destroy(IntPtr handle);
        #endregion

        #region detector.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_detector_create(IntPtr model, String device_name,
            int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_detector_create_by_path(String model_path,
            String device_name, int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_detector_apply(IntPtr handle, mm_mat_t* mats,
            int mat_count, mm_detect_t** results, int** result_count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void c_mmdeploy_detector_release_result(mm_detect_t* results,
            int* result_count, int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void c_mmdeploy_detector_destroy(IntPtr handle);
        #endregion

        #region restorer.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_restorer_create(IntPtr model, String device_name,
            int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_restorer_create_by_path(String model_path,
            String device_name, int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_restorer_apply(IntPtr handle, mm_mat_t* images,
            int count, mm_mat_t** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void c_mmdeploy_restorer_release_result(mm_mat_t* results,
            int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void c_mmdeploy_restorer_destroy(IntPtr handle);
        #endregion

        #region segmentor.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_segmentor_create(IntPtr model, String device_name,
            int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_segmentor_create_by_path(String model_path,
            String device_name, int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_segmentor_apply(IntPtr handle, mm_mat_t* mats,
            int mat_count, mm_segment_t** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void c_mmdeploy_segmentor_release_result(mm_segment_t* results,
            int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void c_mmdeploy_segmentor_destroy(IntPtr handle);
        #endregion

        #region text_detector.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_text_detector_create(IntPtr model, String device_name,
            int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_text_detector_create_by_path(String model_path,
            String device_name, int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_text_detector_apply(IntPtr handle, mm_mat_t* mats,
            int mat_count, mm_text_detect_t** results, int** result_count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void c_mmdeploy_text_detector_release_result(mm_text_detect_t* results,
            int* result_count, int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void c_mmdeploy_text_detector_destroy(IntPtr handle);
        #endregion

        #region text_recognizer.h
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_text_recognizer_create(IntPtr model, String device_name,
            int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int c_mmdeploy_text_recognizer_create_by_path(String model_path,
            String device_name, int device_id, out IntPtr handle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_text_recognizer_apply(IntPtr handle, mm_mat_t* images,
            int count, mm_text_recognize_t** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe int c_mmdeploy_text_recognizer_apply_bbox(IntPtr handle,
            mm_mat_t* images, int image_count, mm_text_detect_t* bboxes, int* bbox_count,
            mm_text_recognize_t** results);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern unsafe void c_mmdeploy_text_recognizer_release_result(
            mm_text_recognize_t* results, int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void c_mmdeploy_text_recognizer_destroy(IntPtr handle);
        #endregion
    }
}
