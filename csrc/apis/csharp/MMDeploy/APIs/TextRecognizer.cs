using System;
using System.Collections.Generic;
using System.Linq;

namespace MMDeploy
{
#pragma warning disable 0649
    internal unsafe struct CTextRecognize
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
    public struct TextRecognize
    {
        /// <summary>
        /// Texts.
        /// </summary>
        public byte[] Text;

        /// <summary>
        /// Scores.
        /// </summary>
        public float[] Score;

        internal unsafe TextRecognize(CTextRecognize* result)
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
        public List<TextRecognize> Results;

        private void Init()
        {
            if (Results == null)
            {
                Results = new List<TextRecognize>();
            }
        }

        internal unsafe void Add(CTextRecognize* result)
        {
            Init();
            Results.Add(new TextRecognize(result));
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
        public List<TextRecognizerOutput> Apply(Mat[] mats)
        {
            List<TextRecognizerOutput> output = new List<TextRecognizerOutput>();
            unsafe
            {
                CTextRecognize* results = null;
                fixed (Mat* _mats = mats)
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
        public List<TextRecognizerOutput> Apply(Mat[] mats, List<TextDetectorOutput> vdetects)
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

                TextDetect[] bboxes = new TextDetect[sz];
                int pos = 0;
                for (int i = 0; i < vdetects.Count; i++)
                {
                    for (int j = 0; j < vdetects[i].Count; j++)
                    {
                        bboxes[pos++] = vdetects[i].Results[j];
                    }
                }

                CTextRecognize* results = null;
                fixed (Mat* _mats = mats)
                fixed (TextDetect* _bboxes = bboxes)
                fixed (int* _bbox_count = bbox_count)
                {
                    ThrowException(NativeMethods.mmdeploy_text_recognizer_apply_bbox(_handle, _mats, mats.Length, _bboxes, _bbox_count, &results));
                    FormatResult(mats.Length, _bbox_count, results, ref output, out var total);
                    ReleaseResult(results, total);
                }
            }

            return output;
        }

        private unsafe void FormatResult(int matCount, int* resultCount, CTextRecognize* results, ref List<TextRecognizerOutput> output, out int total)
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

        private unsafe void ReleaseResult(CTextRecognize* results, int count)
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
