using System;
using System.Collections.Generic;

namespace MMDeploy
{
    /// <summary>
    /// Box.
    /// </summary>
    public struct TextBox
    {
        /// <summary>
        /// P1.
        /// </summary>
        public Pointf P1;

        /// <summary>
        /// P2.
        /// </summary>
        public Pointf P2;

        /// <summary>
        /// P3.
        /// </summary>
        public Pointf P3;

        /// <summary>
        /// P4.
        /// </summary>
        public Pointf P4;

        /// <summary>
        /// Get reference Pi.
        /// </summary>
        /// <param name="i">ith point.</param>
        /// <returns>Pi reference.</returns>
        public Pointf this[int i]
        {
            readonly get
            {
                return i switch
                {
                    0 => P1,
                    1 => P2,
                    2 => P3,
                    3 => P4,
                    _ => throw new ArgumentOutOfRangeException(nameof(i))
                };
            }
            set
            {
                switch (i)
                {
                    case 0: P1 = value; break;
                    case 1: P2 = value; break;
                    case 2: P3 = value; break;
                    case 3: P4 = value; break;
                    default: throw new ArgumentOutOfRangeException(nameof(i));
                }
            }
        }
    }

    /// <summary>
    /// Single detection result of a picture.
    /// A picture may contains multiple reuslts.
    /// </summary>
    public struct TextDetect
    {
        /// <summary>
        /// Bounding box.
        /// </summary>
        public TextBox BBox;

        /// <summary>
        /// Score.
        /// </summary>
        public float Score;

        /// <summary>
        /// Initializes a new instance of the <see cref="TextDetect"/> struct.
        /// </summary>
        /// <param name="score">score.</param>
        /// <param name="bbox">bbox.</param>
        public TextDetect(TextBox bbox, float score)
        {
            BBox = bbox;
            Score = score;
        }

        internal unsafe TextDetect(TextDetect* result)
        {
            Score = result->Score;
            BBox = default;
            for (int i = 0; i < 4; i++)
            {
                BBox[i] = result->BBox[i];
            }
        }
    }

    /// <summary>
    /// Output of DetectorOutput.
    /// </summary>
    public struct TextDetectorOutput
    {
        /// <summary>
        /// Detection results for single image.
        /// </summary>
        public List<TextDetect> Results;

        private void Init()
        {
            if (Results == null)
            {
                Results = new List<TextDetect>();
            }
        }

        /// <summary>
        /// Add result to single image.
        /// </summary>
        /// <param name="bbox">bbox.</param>
        /// <param name="score">score.</param>
        public void Add(TextBox bbox, float score)
        {
            Init();
            Results.Add(new TextDetect(bbox, score));
        }

        internal unsafe void Add(TextDetect* result)
        {
            Init();
            Results.Add(new TextDetect(result));
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
    /// TextDetector.
    /// </summary>
    public class TextDetector : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TextDetector"/> class.
        /// </summary>
        /// <param name="modelPath">model path.</param>
        /// <param name="deviceName">device name.</param>
        /// <param name="deviceId">device id.</param>
        public TextDetector(string modelPath, string deviceName, int deviceId)
        {
            ThrowException(NativeMethods.mmdeploy_text_detector_create_by_path(modelPath, deviceName, deviceId, out _handle));
        }

        /// <summary>
        /// Get information of each image in a batch.
        /// </summary>
        /// <param name="mats">input mats.</param>
        /// <returns>Results of each input mat.</returns>
        public List<TextDetectorOutput> Apply(Mat[] mats)
        {
            List<TextDetectorOutput> output = new List<TextDetectorOutput>();
            unsafe
            {
                TextDetect* results = null;
                int* resultCount = null;
                fixed (Mat* _mats = mats)
                {
                    ThrowException(NativeMethods.mmdeploy_text_detector_apply(_handle, _mats, mats.Length, &results, &resultCount));
                }

                FormatResult(mats.Length, resultCount, results, ref output, out var total);
                ReleaseResult(results, resultCount, total);
            }

            return output;
        }

        private unsafe void FormatResult(int matCount, int* resultCount, TextDetect* results, ref List<TextDetectorOutput> output, out int total)
        {
            total = matCount;
            for (int i = 0; i < matCount; i++)
            {
                TextDetectorOutput outi = default;
                for (int j = 0; j < resultCount[i]; j++)
                {
                    outi.Add(results);
                    results++;
                }

                output.Add(outi);
            }
        }

        private unsafe void ReleaseResult(TextDetect* results, int* resultCount, int count)
        {
            NativeMethods.mmdeploy_text_detector_release_result(results, resultCount, count);
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_text_detector_destroy(_handle);
        }
    }
}
