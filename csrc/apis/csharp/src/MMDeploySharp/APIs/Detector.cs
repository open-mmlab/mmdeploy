using System;
using System.Collections.Generic;

namespace MMDeploySharp
{
#pragma warning disable 0649
    /// <summary>
    /// mm_instance_mask_t of c code.
    /// </summary>
    internal unsafe struct CMmInstanceMask
    {
        public char* Data;
        public int Height;
        public int Width;
    }

    /// <summary>
    /// mm_detect_t of c code.
    /// </summary>
    internal unsafe struct CMmDetect
    {
        public int LabelId;
        public float Score;
        public MmRect BBox;
        public CMmInstanceMask* Mask;
    }
#pragma warning restore 0649

    /// <summary>
    /// Instance mask.
    /// </summary>
    public struct MmInstanceMask
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
        internal unsafe MmInstanceMask(CMmInstanceMask* mask)
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
    public struct MmDetect
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
        /// Bouding box.
        /// </summary>
        public MmRect BBox;

        /// <summary>
        /// Whether has mask.
        /// </summary>
        public bool HasMask;

        /// <summary>
        /// Mask.
        /// </summary>
        public MmInstanceMask Mask;

        /// <summary>
        /// Initializes a new instance of the <see cref="MmDetect"/> struct.
        /// </summary>
        /// <param name="labelId">label id.</param>
        /// <param name="score"> score.</param>
        /// <param name="bbox">bounding box.</param>
        public MmDetect(int labelId, float score, MmRect bbox)
        {
            LabelId = labelId;
            Score = score;
            BBox = bbox;
            HasMask = false;
            Mask = default;
        }

        internal unsafe MmDetect(CMmDetect* result) : this(result->LabelId, result->Score, result->BBox)
        {
            if (result->Mask != null)
            {
                HasMask = true;
                CMmInstanceMask* mask = result->Mask;
                Mask = new MmInstanceMask(mask);
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
        public List<MmDetect> Results;

        /// <summary>
        /// Init Reuslts.
        /// </summary>
        public void Init()
        {
            if (Results == null)
            {
                Results = new List<MmDetect>();
            }
        }

        /// <summary>
        /// Add result to single image.
        /// </summary>
        /// <param name="labelId">label id.</param>
        /// <param name="score">score.</param>
        /// <param name="bbox">bounding box.</param>
        public void Add(int labelId, float score, MmRect bbox)
        {
            Init();
            Results.Add(new MmDetect(labelId, score, bbox));
        }

        internal unsafe void Add(CMmDetect* result)
        {
            Init();
            Results.Add(new MmDetect(result));
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
        public List<DetectorOutput> Apply(MmMat[] mats)
        {
            List<DetectorOutput> output = new List<DetectorOutput>();

            unsafe
            {
                CMmDetect* results = null;
                int* resultCount = null;
                fixed (MmMat* _mats = mats)
                {
                    ThrowException(NativeMethods.mmdeploy_detector_apply(_handle, _mats, mats.Length, &results, &resultCount));
                }

                FormatResult(mats.Length, resultCount, results, ref output, out var total);
                ReleaseResult(results, resultCount, total);
            }

            return output;
        }

        private unsafe void FormatResult(int matCount, int* resultCount, CMmDetect* results, ref List<DetectorOutput> output, out int total)
        {
            total = 0;
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

        private unsafe void ReleaseResult(CMmDetect* results, int* resultCount, int count)
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
