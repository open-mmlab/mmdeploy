using System;
using System.Collections.Generic;

namespace MMDeploy
{
    /// <summary>
    /// Single detection result of a picture.
    /// A picture may contains multiple reuslts.
    /// </summary>
    public struct RDetect
    {
        /// <summary>
        /// Label id.
        /// </summary>
        public int LabelId;

        /// <summary>
        /// Score.
        /// </summary>
        public float Score;

        /// <summary>
        /// Center x.
        /// </summary>
        public float Cx;

        /// <summary>
        /// Center y.
        /// </summary>
        public float Cy;

        /// <summary>
        /// Width.
        /// </summary>
        public float Width;

        /// <summary>
        /// Height.
        /// </summary>
        public float Height;

        /// <summary>
        /// Angle.
        /// </summary>
        public float Angle;

        internal unsafe RDetect(RDetect* result) : this()
        {
            this = *result;
        }
    }

    /// <summary>
    /// Output of RotatedDetector.
    /// </summary>
    public struct RotatedDetectorOutput
    {
        /// <summary>
        /// Rotated detection results for single image.
        /// </summary>
        public List<RDetect> Results;

        private void Init()
        {
            if (Results == null)
            {
                Results = new List<RDetect>();
            }
        }

        internal unsafe void Add(RDetect* result)
        {
            Init();
            Results.Add(new RDetect(result));
        }

        /// <summary>
        /// Gets number of output.
        /// </summary>
        public int Count
        {
            get { return (Results == null) ? 0 : Results.Count; }
        }
    }

    /// <summary>
    /// RotatedDetector.
    /// </summary>
    public class RotatedDetector : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RotatedDetector"/> class.
        /// </summary>
        /// <param name="modelPath">model path.</param>
        /// <param name="deviceName">device name.</param>
        /// <param name="deviceId">device id.</param>
        public RotatedDetector(string modelPath, string deviceName, int deviceId)
        {
            ThrowException(NativeMethods.mmdeploy_rotated_detector_create_by_path(modelPath,
                deviceName, deviceId, out _handle));
        }

        /// <summary>
        /// Get information of each image in a batch.
        /// </summary>
        /// <param name="mats">input mats.</param>
        /// <returns>Results of each input mat.</returns>
        public List<RotatedDetectorOutput> Apply(Mat[] mats)
        {
            List<RotatedDetectorOutput> output = new List<RotatedDetectorOutput>();

            unsafe
            {
                RDetect* results = null;
                int* resultCount = null;
                fixed (Mat* _mats = mats)
                {
                    ThrowException(NativeMethods.mmdeploy_rotated_detector_apply(_handle,
                        _mats, mats.Length, &results, &resultCount));
                }

                FormatResult(mats.Length, resultCount, results, ref output, out var total);
                ReleaseResult(results, resultCount);
            }

            return output;
        }

        private unsafe void FormatResult(int matCount, int* resultCount, RDetect* results,
            ref List<RotatedDetectorOutput> output, out int total)
        {
            total = matCount;
            for (int i = 0; i < matCount; i++)
            {
                RotatedDetectorOutput outi = default;
                for (int j = 0; j < resultCount[i]; j++)
                {
                    outi.Add(results);
                    results++;
                }

                output.Add(outi);
            }
        }

        private unsafe void ReleaseResult(RDetect* results, int* resultCount)
        {
            NativeMethods.mmdeploy_rotated_detector_release_result(results, resultCount);
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_rotated_detector_destroy(_handle);
        }
    }
}
