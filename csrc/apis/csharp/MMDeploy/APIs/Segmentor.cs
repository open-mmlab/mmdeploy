using System;
using System.Collections.Generic;

namespace MMDeploy
{
#pragma warning disable 0649
    internal unsafe struct CSegment
    {
        public int Height;
        public int Width;
        public int Classes;
        public int* Mask;
    }
#pragma warning restore 0649

    /// <summary>
    /// Output of Segmentor.
    /// </summary>
    public struct SegmentorOutput
    {
        /// <summary>
        /// Height of image.
        /// </summary>
        public int Height;

        /// <summary>
        /// Width if image.
        /// </summary>
        public int Width;

        /// <summary>
        /// Number of classes.
        /// </summary>
        public int Classes;

        /// <summary>
        /// Mask data.
        /// </summary>
        public int[] Mask;

        /// <summary>
        /// Initializes a new instance of the <see cref="SegmentorOutput"/> struct.
        /// </summary>
        /// <param name="height">height.</param>
        /// <param name="width">width.</param>
        /// <param name="classes">classes.</param>
        /// <param name="mask">mask.</param>
        public SegmentorOutput(int height, int width, int classes, int[] mask)
        {
            Height = height;
            Width = width;
            Classes = classes;
            Mask = new int[Height * Width];
            Array.Copy(mask, this.Mask, mask.Length);
        }

        internal unsafe SegmentorOutput(CSegment* result)
        {
            Height = result->Height;
            Width = result->Width;
            Classes = result->Classes;
            Mask = new int[Height * Width];
            int nbytes = Height * Width * sizeof(int);
            fixed (int* data = this.Mask)
            {
                Buffer.MemoryCopy(result->Mask, data, nbytes, nbytes);
            }
        }
    }

    /// <summary>
    /// Segmentor.
    /// </summary>
    public class Segmentor : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Segmentor"/> class.
        /// </summary>
        /// <param name="modelPath">model path.</param>
        /// <param name="deviceName">device name.</param>
        /// <param name="deviceId">device id.</param>
        public Segmentor(string modelPath, string deviceName, int deviceId)
        {
            ThrowException(NativeMethods.mmdeploy_segmentor_create_by_path(modelPath, deviceName, deviceId, out _handle));
        }

        /// <summary>
        /// Get information of each image in a batch.
        /// </summary>
        /// <param name="mats">input mats.</param>
        /// <returns>Results of each input mat.</returns>
        public List<SegmentorOutput> Apply(Mat[] mats)
        {
            List<SegmentorOutput> output = new List<SegmentorOutput>();
            unsafe
            {
                CSegment* results = null;
                fixed (Mat* _mats = mats)
                {
                    ThrowException(NativeMethods.mmdeploy_segmentor_apply(_handle, _mats, mats.Length, &results));
                }

                FormatResult(mats.Length, results, ref output, out var total);
                ReleaseResult(results, total);
            }

            return output;
        }

        private unsafe void FormatResult(int matCount, CSegment* results, ref List<SegmentorOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < matCount; i++)
            {
                SegmentorOutput outi = new SegmentorOutput(results);
                results++;
                total++;
                output.Add(outi);
            }
        }

        private unsafe void ReleaseResult(CSegment* results, int count)
        {
            NativeMethods.mmdeploy_segmentor_release_result(results, count);
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_segmentor_destroy(_handle);
        }
    }
}
