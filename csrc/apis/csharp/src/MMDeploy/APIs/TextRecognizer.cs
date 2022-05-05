using System;
using System.Collections.Generic;
using System.Linq;

namespace MMDeploy
{
#pragma warning disable 0649
    internal unsafe struct CMmTextRecognize
    {
        public char* Text;
        public float* Score;
        public int Length;
    }
#pragma warning restore 0649

    /// <summary>
    /// Single result of a picture.
    /// A picture may contains multiple reuslts.
    /// </summary>
    public struct MmTextRecognize
    {
        /// <summary>
        /// Texts.
        /// </summary>
        public byte[] Text;

        /// <summary>
        /// Scores.
        /// </summary>
        public float[] Score;

        internal unsafe MmTextRecognize(CMmTextRecognize* result)
        {
            Text = new byte[result->Length];
            Score = new float[result->Length];
            fixed (byte* _text = Text)
            {
                int nbytes = result->Length;
                Buffer.MemoryCopy(result->Text, _text, nbytes, nbytes);
            }

            fixed (float* _score = Score)
            {
                int nbytes = result->Length * sizeof(float);
                Buffer.MemoryCopy(result->Score, _score, nbytes, nbytes);
            }
        }
    }

    /// <summary>
    /// Output of TextRecognizer.
    /// </summary>
    public struct TextRecognizerOutput
    {
        /// <summary>
        /// Text recognization results for single image.
        /// </summary>
        public List<MmTextRecognize> Results;

        private void Init()
        {
            if (Results == null)
            {
                Results = new List<MmTextRecognize>();
            }
        }

        internal unsafe void Add(CMmTextRecognize* result)
        {
            Init();
            Results.Add(new MmTextRecognize(result));
        }
    }

    /// <summary>
    /// TextRecognizer.
    /// </summary>
    public class TextRecognizer : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TextRecognizer"/> class.
        /// </summary>
        /// <param name="modelPath">model path.</param>
        /// <param name="deviceName">device name.</param>
        /// <param name="deviceId">device id.</param>
        public TextRecognizer(string modelPath, string deviceName, int deviceId)
        {
            ThrowException(NativeMethods.mmdeploy_text_recognizer_create_by_path(modelPath, deviceName, deviceId, out _handle));
        }

        /// <summary>
        /// Get information of each image in a batch.
        /// </summary>
        /// <param name="mats">input mats.</param>
        /// <returns>Results of each input mat.</returns>
        public List<TextRecognizerOutput> Apply(MmMat[] mats)
        {
            List<TextRecognizerOutput> output = new List<TextRecognizerOutput>();
            unsafe
            {
                CMmTextRecognize* results = null;
                fixed (MmMat* _mats = mats)
                {
                    ThrowException(NativeMethods.mmdeploy_text_recognizer_apply(_handle, _mats, mats.Length, &results));
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

        /// <summary>
        /// Get information of each image in a batch.
        /// </summary>
        /// <param name="mats">input mats.</param>
        /// <param name="vdetects">detection for each image.</param>
        /// <returns>Results of each input mat.</returns>
        public List<TextRecognizerOutput> Apply(MmMat[] mats, List<TextDetectorOutput> vdetects)
        {
            List<TextRecognizerOutput> output = new List<TextRecognizerOutput>();
            unsafe
            {
                int[] bbox_count = new int[vdetects.Count];
                int sz = 0;
                for (int i = 0; i < vdetects.Count; i++)
                {
                    bbox_count[i] = vdetects[i].Count;
                    sz += bbox_count[i];
                }

                MmTextDetect[] bboxes = new MmTextDetect[sz];
                int pos = 0;
                for (int i = 0; i < vdetects.Count; i++)
                {
                    for (int j = 0; j < vdetects[i].Count; j++)
                    {
                        bboxes[pos++] = vdetects[i].Results[j];
                    }
                }

                CMmTextRecognize* results = null;
                fixed (MmMat* _mats = mats)
                fixed (MmTextDetect* _bboxes = bboxes)
                fixed (int* _bbox_count = bbox_count)
                {
                    ThrowException(NativeMethods.mmdeploy_text_recognizer_apply_bbox(_handle, _mats, mats.Length, _bboxes, _bbox_count, &results));
                    FormatResult(mats.Length, _bbox_count, results, ref output, out var total);
                    ReleaseResult(results, total);
                }
            }

            return output;
        }

        private unsafe void FormatResult(int matCount, int* resultCount, CMmTextRecognize* results, ref List<TextRecognizerOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < matCount; i++)
            {
                TextRecognizerOutput outi = default;
                for (int j = 0; j < resultCount[i]; j++)
                {
                    outi.Add(results);
                    results++;
                    total++;
                }

                output.Add(outi);
            }
        }

        private unsafe void ReleaseResult(CMmTextRecognize* results, int count)
        {
            NativeMethods.mmdeploy_text_recognizer_release_result(results, count);
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_text_recognizer_destroy(_handle);
        }
    }
}
