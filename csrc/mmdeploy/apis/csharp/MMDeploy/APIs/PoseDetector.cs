using System.Collections.Generic;
using System.Linq;

namespace MMDeploy
{
#pragma warning disable 0649
    internal unsafe struct CPoseDetect
    {
        public Pointf* Point;
        public float* Score;
        public int Length;
    }
#pragma warning restore 0649

    /// <summary>
    /// Single detection result of a bbox.
    /// A picture may contains multiple reuslts.
    /// </summary>
    public struct PoseDetect
    {
        /// <summary>
        /// Keypoins.
        /// </summary>
        public List<Pointf> Points;

        /// <summary>
        /// Scores.
        /// </summary>
        public List<float> Scores;

        /// <summary>
        /// Init points and scores if empty.
        /// </summary>
        private void Init()
        {
            if (Points == null || Scores == null)
            {
                Points = new List<Pointf>();
                Scores = new List<float>();
            }
        }

        /// <summary>
        /// Add single keypoint to list.
        /// </summary>
        /// <param name="point">Keypoint.</param>
        /// <param name="score">Score.</param>
        public void Add(Pointf point, float score)
        {
            Init();
            Points.Add(point);
            Scores.Add(score);
        }

        internal unsafe void Add(Pointf* point, float score)
        {
            Init();
            Points.Add(new Pointf(point->X, point->Y));
        }
    }

    /// <summary>
    /// Output of PoseDetector.
    /// </summary>
    public struct PoseDetectorOutput
    {
        /// <summary>
        /// Pose detection results for single image.
        /// </summary>
        public List<PoseDetect> Results;

        /// <summary>
        /// Gets number of output.
        /// </summary>
        public int Count
        {
            get { return (Results == null) ? 0 : Results.Count; }
        }

        /// <summary>
        /// Result for box level.
        /// </summary>
        /// <param name="boxRes">Box res.</param>
        public void Add(PoseDetect boxRes)
        {
            if (Results == null)
            {
                Results = new List<PoseDetect>();
            }

            Results.Add(boxRes);
        }
    }

    /// <summary>
    /// PoseDetector.
    /// </summary>
    public class PoseDetector : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PoseDetector"/> class.
        /// </summary>
        /// <param name="modelPath">model path.</param>
        /// <param name="deviceName">device name.</param>
        /// <param name="deviceId">device id.</param>
        public PoseDetector(string modelPath, string deviceName, int deviceId)
        {
            ThrowException(NativeMethods.mmdeploy_pose_detector_create_by_path(modelPath, deviceName, deviceId, out _handle));
        }

        /// <summary>
        /// Get information of each image in a batch.
        /// </summary>
        /// <param name="mats">input mats.</param>
        /// <param name="bboxes">bounding boxes..</param>
        /// <param name="bboxCount">bounding boxes count for each image.</param>
        /// <returns>Results of each input mat.</returns>
        public List<PoseDetectorOutput> Apply(Mat[] mats, Rect[] bboxes, int[] bboxCount)
        {
            List<PoseDetectorOutput> output = new List<PoseDetectorOutput>();

            unsafe
            {
                CPoseDetect* results = null;
                fixed (Mat* _mats = mats)
                fixed (Rect* _bboxes = bboxes)
                fixed (int* _bboxCount = bboxCount)
                {
                    ThrowException(NativeMethods.mmdeploy_pose_detector_apply_bbox(_handle, _mats, mats.Length, _bboxes, _bboxCount, &results));
                    FormatResult(mats.Length, _bboxCount, results, ref output, out var total);
                    ReleaseResult(results, total);
                }
            }

            return output;
        }

        /// <summary>
        /// Get information of each image in a batch.
        /// </summary>
        /// <param name="mats">Input mats.</param>
        /// <returns>Results of each input mat.</returns>
        public List<PoseDetectorOutput> Apply(Mat[] mats)
        {
            List<PoseDetectorOutput> output = new List<PoseDetectorOutput>();
            unsafe
            {
                CPoseDetect* results = null;
                fixed (Mat* _mats = mats)
                {
                    ThrowException(NativeMethods.mmdeploy_pose_detector_apply(_handle, _mats, mats.Length, &results));
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

        private unsafe void FormatResult(int matCount, int* bboxCount, CPoseDetect* results, ref List<PoseDetectorOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < matCount; i++)
            {
                PoseDetectorOutput outi = default;
                for (int j = 0; j < bboxCount[i]; j++)
                {
                    PoseDetect boxRes = default;
                    for (int k = 0; k < results->Length; k++)
                    {
                        boxRes.Add(results->Point[k], results->Score[k]);
                    }

                    outi.Add(boxRes);
                    results++;
                    total++;
                }

                output.Add(outi);
            }
        }

        private unsafe void ReleaseResult(CPoseDetect* results, int count)
        {
            NativeMethods.mmdeploy_pose_detector_release_result(results, count);
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_pose_detector_destroy(_handle);
        }
    }
}
