using System;
using System.Collections.Generic;
using System.Linq;

#pragma warning disable CS8602

namespace MMDeploySharp
{
    unsafe public struct mm_pose_detect_t
    {
        public mm_pointf_t* point;  ///< keypoint
        public float* score;        ///< keypoint score
        public int length;          ///< number of keypoint
    }

    public struct KeyPoints2D
    {
        public struct Point
        {
            public Point(float x, float y, float z)
            {
                this.x = x;
                this.y = y;
                this.z = z;
            }

            public float x;
            public float y;
            public float z;
        }

        public void Add(float x, float y, float z)
        {
            if (points == null)
            {
                points = new List<Point>();
            }
            points.Add(new Point(x, y, z));
        }

        public List<Point> points;
        public int Count
        {
            get { return (points == null) ? 0 : points.Count; }
        }
    }

    public struct PoseDetectorOutput
    {
        public List<KeyPoints2D> boxes;
        public int Count
        {
            get { return (boxes == null) ? 0 : boxes.Count; }
        }

        public void Add(KeyPoints2D box_res)
        {
            if (boxes == null)
            {
                boxes = new List<KeyPoints2D>();
            }
            boxes.Add(box_res);
        }
    }

    public class PoseDetector : DisposableObject
    {
        public PoseDetector(String model_path, String device_name, int device_id)
        {
            ThrowException(NativeMethods.c_mmdeploy_pose_detector_create_by_path(model_path, device_name, device_id, out _handle));
        }

        public List<PoseDetectorOutput> Apply(mm_mat_t[] mats, mm_rect_t[] bboxes, int[] bbox_count)
        {
            List<PoseDetectorOutput> output = new List<PoseDetectorOutput>();

            unsafe
            {
                mm_pose_detect_t* results = null;
                fixed (mm_mat_t* _mats = mats)
                fixed (mm_rect_t* _bboxes = bboxes)
                fixed (int* _bbox_count = bbox_count)
                {
                    ThrowException(NativeMethods.c_mmdeploy_pose_detector_apply_bbox(_handle, _mats, mats.Length, _bboxes, _bbox_count, &results));
                    FormatResult(mats.Length, _bbox_count, results, ref output, out var total);
                    ReleaseResult(results, total);
                }
            }
            return output;
        }

        public List<PoseDetectorOutput> Apply(mm_mat_t[] mats)
        {
            List<PoseDetectorOutput> output = new List<PoseDetectorOutput>();
            unsafe
            {
                mm_pose_detect_t* results = null;
                fixed (mm_mat_t* _mats = mats)
                {
                    ThrowException(NativeMethods.c_mmdeploy_pose_detector_apply(_handle, _mats, mats.Length, &results));
                }

                int[] _bbox_count = Enumerable.Repeat(1, mats.Length).ToArray();
                fixed (int* bbox_count = _bbox_count)
                {
                    FormatResult(mats.Length, bbox_count, results, ref output, out var total);
                    ReleaseResult(results, total);
                }
            }

            return output;
        }

        public unsafe void FormatResult(int mat_count, int* bbox_count, mm_pose_detect_t* results, ref List<PoseDetectorOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < mat_count; i++)
            {
                PoseDetectorOutput outi = new PoseDetectorOutput();
                for (int j = 0; j < bbox_count[i]; j++)
                {
                    KeyPoints2D box_res = new KeyPoints2D();
                    for (int k = 0; k < results->length; k++)
                    {
                        box_res.Add(results->point[k].x, results->point[k].y, results->score[k]);
                    }
                    outi.Add(box_res);
                    results++;
                    total++;
                }
                output.Add(outi);
            }
        }

        public unsafe void ReleaseResult(mm_pose_detect_t* results, int count)
        {
            NativeMethods.c_mmdeploy_pose_detector_release_result(results, count);
        }

        protected override void ReleaseHandle()
        {
            NativeMethods.c_mmdeploy_pose_detector_destroy(_handle);
        }
    }
}
