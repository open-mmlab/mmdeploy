using System;
using System.Collections.Generic;

namespace MMDeploy
{
    /// <summary>
    /// Output of Restorer.
    /// </summary>
    public struct RestorerOutput
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

        /// <summary>
        /// Initializes a new instance of the <see cref="RestorerOutput"/> struct.
        /// </summary>
        /// <param name="height">height.</param>
        /// <param name="width">width.</param>
        /// <param name="data">data.</param>
        public RestorerOutput(int height, int width, byte[] data)
        {
            Height = height;
            Width = width;
            Data = new byte[height * width * 3];
            Array.Copy(data, Data, data.Length);
        }

        internal unsafe RestorerOutput(Mat* result)
        {
            Height = result->Height;
            Width = result->Width;
            Data = new byte[Height * Width * 3];
            int nbytes = Height * Width * 3;
            fixed (byte* data = this.Data)
            {
                Buffer.MemoryCopy(result->Data, data, nbytes, nbytes);
            }
        }
    }

    /// <summary>
    /// Restorer.
    /// </summary>
    public class Restorer : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Restorer"/> class.
        /// </summary>
        /// <param name="modelPath">model path.</param>
        /// <param name="deviceName">device name.</param>
        /// <param name="deviceId">device id.</param>
        public Restorer(string modelPath, string deviceName, int deviceId)
        {
            ThrowException(NativeMethods.mmdeploy_restorer_create_by_path(modelPath, deviceName, deviceId, out _handle));
        }

        /// <summary>
        /// Get information of each image in a batch.
        /// </summary>
        /// <param name="mats">input mats.</param>
        /// <returns>Results of each input mat.</returns>
        public List<RestorerOutput> Apply(Mat[] mats)
        {
            List<RestorerOutput> output = new List<RestorerOutput>();
            unsafe
            {
                Mat* results = null;
                fixed (Mat* _mats = mats)
                {
                    ThrowException(NativeMethods.mmdeploy_restorer_apply(_handle, _mats, mats.Length, &results));
                }

                FormatResult(mats.Length, results, ref output, out var total);
                ReleaseResult(results, total);
            }

            return output;
        }

        private unsafe void FormatResult(int matCount, Mat* results, ref List<RestorerOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < matCount; i++)
            {
                output.Add(new RestorerOutput(results));
                results++;
                total++;
            }
        }

        private unsafe void ReleaseResult(Mat* results, int count)
        {
            NativeMethods.mmdeploy_restorer_release_result(results, count);
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_restorer_destroy(_handle);
        }
    }
}
