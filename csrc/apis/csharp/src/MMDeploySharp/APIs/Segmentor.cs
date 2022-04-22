using System;
using System.Collections.Generic;

namespace MMDeploySharp
{
    unsafe public struct mm_segment_t
    {
        public int height;
        public int width;
        public int classes;
        public int* mask;
    }

    public struct SegmentorOutput
    {
        public int height;
        public int width;
        public int classes;
        public int[] mask;

        public SegmentorOutput(int height, int width, int classes, int[] mask)
        {
            this.height = height;
            this.width = width;
            this.classes = classes;
            this.mask = new int[height * width];
            Array.Copy(mask, this.mask, mask.Length);
        }

        public unsafe SegmentorOutput(mm_segment_t* result)
        {
            this.height = result->height;
            this.width = result->width;
            this.classes = result->classes;
            this.mask = new int[height * width];
            int nbytes = height * width * sizeof(int);
            fixed (int* data = this.mask)
            {
                Buffer.MemoryCopy(result->mask, data, nbytes, nbytes);
            }
        }
    }

    public class Segmentor : DisposableObject
    {
        public Segmentor(String model_path, String device_name, int device_id)
        {
            ThrowException(NativeMethods.c_mmdeploy_segmentor_create_by_path(model_path, device_name, device_id, out _handle));
        }

        public List<SegmentorOutput> Apply(mm_mat_t[] mats)
        {
            List<SegmentorOutput> output = new List<SegmentorOutput>();
            unsafe
            {
                mm_segment_t* results = null;
                fixed (mm_mat_t* _mats = mats)
                {
                    ThrowException(NativeMethods.c_mmdeploy_segmentor_apply(_handle, _mats, mats.Length, &results));
                }

                FormatResult(mats.Length, results, ref output, out var total);
                ReleaseResult(results, total);
            }

            return output;
        }

        public unsafe void FormatResult(int mat_count, mm_segment_t* results, ref List<SegmentorOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < mat_count; i++)
            {
                SegmentorOutput outi = new SegmentorOutput(results);
                results++;
                output.Add(outi);
            }
        }

        public unsafe void ReleaseResult(mm_segment_t* results, int count)
        {
            NativeMethods.c_mmdeploy_segmentor_release_result(results, count);
        }

        protected override void ReleaseHandle()
        {
            NativeMethods.c_mmdeploy_segmentor_destroy(_handle);
        }
    }
}
