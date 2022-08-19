using System;
using System.Collections.Generic;

namespace MMDeploy
{
#pragma warning disable 0649
    /// <summary>
    /// mm_instance_mask_t of c code.
    /// </summary>
    internal unsafe struct CInstanceMask
    {
        public char* Data;
        public int Height;
        public int Width;
    }

    /// <summary>
    /// mm_detect_t of c code.
    /// </summary>
    internal unsafe struct CDetect
    {
        public int LabelId;
        public float Score;
        public Rect BBox;
        public CInstanceMask* Mask;
    }
#pragma warning restore 0649

    /// <summary>
    /// Instance mask.
    /// </summary>
    public struct InstanceMask
    {
        /// <summary>
        /// Height.
        /// </summary>
        public int Height;

        /// <summary>
        /// Width.
        /// </summary>
        public int Width;

        /// <summary>
        /// Raw data.
        /// </summary>
        public byte[] Data;
        internal unsafe InstanceMask(CInstanceMask* mask)
        {
            Height = mask->Height;
            Width = mask->Width;
            Data = new byte[Height * Width];
            fixed (byte* data = this.Data)
            {
                Buffer.MemoryCopy(mask->Data, data, Height * Width, Height * Width);
            }
        }
    }

    /// <summary>
    /// Single detection result of a picture.
    /// A picture may contains multiple reuslts.
    /// </summary>
    public struct Detect
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
        /// Bounding box.
        /// </summary>
        public Rect BBox;

        /// <summary>
        /// Whether has mask.
        /// </summary>
        public bool HasMask;

        /// <summary>
        /// Mask.
        /// </summary>
        public InstanceMask Mask;

        /// <summary>
        /// Initializes a new instance of the <see cref="Detect"/> struct.
        /// </summary>
        /// <param name="labelId">label id.</param>
        /// <param name="score"> score.</param>
        /// <param name="bbox">bounding box.</param>
        public Detect(int labelId, float score, Rect bbox)
        {
            LabelId = labelId;
            Score = score;
            BBox = bbox;
            HasMask = false;
            Mask = default;
        }

        internal unsafe Detect(CDetect* result) : this(result->LabelId, result->Score, result->BBox)
        {
            if (result->Mask != null)
            {
                HasMask = true;
                CInstanceMask* mask = result->Mask;
                Mask = new InstanceMask(mask);
            }
        }
    }

    /// <summary>
    /// Output of Detector.
    /// </summary>
    public struct DetectorOutput
    {
        /// <summary>
        /// Detection results for single image.
        /// </summary>
        public List<Detect> Results;

        /// <summary>
        /// Init Reuslts.
        /// </summary>
        public void Init()
        {
            if (Results == null)
            {
                Results = new List<Detect>();
            }
        }

        /// <summary>
        /// Add result to single image.
        /// </summary>
        /// <param name="labelId">label id.</param>
        /// <param name="score">score.</param>
        /// <param name="bbox">bounding box.</param>
        public void Add(int labelId, float score, Rect bbox)
        {
            Init();
            Results.Add(new Detect(labelId, score, bbox));
        }

        internal unsafe void Add(CDetect* result)
        {
            Init();
            Results.Add(new Detect(result));
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
    /// Detector.
    /// </summary>
    public class Detector : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Detector"/> class.
        /// </summary>
        /// <param name="modelPath">model path.</param>
        /// <param name="deviceName">device name.</param>
        /// <param name="deviceId">device id.</param>
        public Detector(string modelPath, string deviceName, int deviceId)
        {
            ThrowException(NativeMethods.mmdeploy_detector_create_by_path(modelPath, deviceName, deviceId, out _handle));
        }

        /// <summary>
        /// Get information of each image in a batch.
        /// </summary>
        /// <param name="mats">input mats.</param>
        /// <returns>Results of each input mat.</returns>
        public List<DetectorOutput> Apply(Mat[] mats)
        {
            List<DetectorOutput> output = new List<DetectorOutput>();

            unsafe
            {
                CDetect* results = null;
                int* resultCount = null;
                fixed (Mat* _mats = mats)
                {
                    ThrowException(NativeMethods.mmdeploy_detector_apply(_handle, _mats, mats.Length, &results, &resultCount));
                }

                FormatResult(mats.Length, resultCount, results, ref output, out var total);
                ReleaseResult(results, resultCount, total);
            }

            return output;
        }

        private unsafe void FormatResult(int matCount, int* resultCount, CDetect* results, ref List<DetectorOutput> output, out int total)
        {
            total = matCount;
            for (int i = 0; i < matCount; i++)
            {
                DetectorOutput outi = default;
                for (int j = 0; j < resultCount[i]; j++)
                {
                    outi.Add(results);
                    results++;
                }

                output.Add(outi);
            }
        }

        private unsafe void ReleaseResult(CDetect* results, int* resultCount, int count)
        {
            NativeMethods.mmdeploy_detector_release_result(results, resultCount, count);
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_detector_destroy(_handle);
        }
    }
}
