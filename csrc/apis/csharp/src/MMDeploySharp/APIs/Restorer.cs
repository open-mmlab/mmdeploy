using System;
using System.Collections.Generic;

namespace MMDeploySharp
{
    public struct RestorerOutput
    {
        public RestorerOutput(int height, int width, byte[] data)
        {
            this.height = height;
            this.width = width;
            this.data = new byte[height * width * 3];
            Array.Copy(data, this.data, data.Length);
        }

        public unsafe RestorerOutput(mm_mat_t* result)
        {
            this.height = result->height;
            this.width = result->width;
            this.data = new byte[height * width * 3];
            int nbytes = this.height * this.width * 3;
            fixed (byte* data = this.data)
            {
                Buffer.MemoryCopy(result->data, data, nbytes, nbytes);
            }
        }

        public int height;
        public int width;
        public byte[] data;
    }

    public class Restorer : DisposableObject
    {
        public Restorer(String model_path, String device_name, int device_id)
        {
            ThrowException(NativeMethods.c_mmdeploy_restorer_create_by_path(model_path, device_name, device_id, out _handle));
        }

        public List<RestorerOutput> Apply(mm_mat_t[] mats)
        {
            List<RestorerOutput> output = new List<RestorerOutput>();
            unsafe
            {
                mm_mat_t* results = null;
                fixed (mm_mat_t* _mats = mats)
                {
                    ThrowException(NativeMethods.c_mmdeploy_restorer_apply(_handle, _mats, mats.Length, &results));
                }

                FormatResult(mats.Length, results, ref output, out var total);
                ReleaseResult(results, total);
            }
            return output;
        }

        public unsafe void FormatResult(int mat_count, mm_mat_t* results, ref List<RestorerOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < mat_count; i++)
            {
                output.Add(new RestorerOutput(results));
                results++;
                total++;
            }
        }


        public unsafe void ReleaseResult(mm_mat_t* results, int count)
        {
            NativeMethods.c_mmdeploy_restorer_release_result(results, count);
        }

        protected override void ReleaseHandle()
        {
            NativeMethods.c_mmdeploy_restorer_destroy(_handle);
        }
    }
}
