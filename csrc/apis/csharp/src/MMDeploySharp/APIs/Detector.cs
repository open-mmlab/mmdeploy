using System;
using System.Collections.Generic;

namespace MMDeploySharp
{
    unsafe public struct mm_instance_mask_t
    {
        public char* data;
        public int height;
        public int width;
    }

    public unsafe struct mm_detect_t
    {
        public int label_id;
        public float score;
        public mm_rect_t bbox;
        public mm_instance_mask_t* mask;
    }

    public struct DetectorOutput
    {
        public struct Mask
        {
            public unsafe Mask(mm_instance_mask_t* mask)
            {
                this.height = mask->height;
                this.width = mask->width;
                this.data = new byte[this.height * this.width];
                fixed (byte* data = this.data)
                {
                    Buffer.MemoryCopy(mask->data, data, this.height * this.width, this.height * this.width);
                }
            }
            public int height { get; set; }
            public int width { get; set; }
            public byte[] data;
        }

        public struct Detect
        {
            public Detect(int label_id, float score, mm_rect_t bbox)
            {
                this.label_id = label_id;
                this.score = score;
                this.bbox = bbox;
                this.has_mask = false;
                this.mask = new Mask();
            }

            public unsafe Detect(mm_detect_t* result) : this(result->label_id, result->score, result->bbox)
            {
                if (result->mask != null)
                {
                    this.has_mask = true;
                    mm_instance_mask_t* mask = result->mask;
                    this.mask = new Mask(mask);
                }
            }

            public int label_id;
            public float score;
            public mm_rect_t bbox;
            public bool has_mask;
            public Mask mask;
        }

        public void Init()
        {
            if (detects == null)
            {
                detects = new List<Detect>();
            }
        }

        public void Add(int label_id, float score, mm_rect_t bbox)
        {
            Init();
            detects.Add(new Detect(label_id, score, bbox));
        }

        public unsafe void Add(mm_detect_t* result)
        {
            Init();
            detects.Add(new Detect(result));
        }

        public int Count
        {
            get { return (detects == null) ? 0 : detects.Count; }
        }

        public List<Detect> detects;
    }

    public class Detector : DisposableObject
    {
        public Detector(String model_path, String device_name, int device_id)
        {
            ThrowException(NativeMethods.c_mmdeploy_detector_create_by_path(model_path, device_name, device_id, out _handle));
        }

        public List<DetectorOutput> Apply(mm_mat_t[] mats)
        {
            List<DetectorOutput> output = new List<DetectorOutput>();

            unsafe
            {
                mm_detect_t* results = null;
                int* result_count = null;
                fixed (mm_mat_t* _mats = mats)
                {
                    ThrowException(NativeMethods.c_mmdeploy_detector_apply(_handle, _mats, mats.Length, &results, &result_count));
                }
                FormatResult(mats.Length, result_count, results, ref output, out var total);
                ReleaseResult(results, result_count, total);
            }

            return output;
        }


        public unsafe void FormatResult(int mat_count, int* result_count, mm_detect_t* results, ref List<DetectorOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < mat_count; i++)
            {
                DetectorOutput outi = new DetectorOutput();
                for (int j = 0; j < result_count[i]; j++)
                {
                    outi.Add(results);
                    results++;
                }
                output.Add(outi);
            }
        }

        public unsafe void ReleaseResult(mm_detect_t* results, int* result_count, int count)
        {
            NativeMethods.c_mmdeploy_detector_release_result(results, result_count, count);
        }

        protected override void ReleaseHandle()
        {
            NativeMethods.c_mmdeploy_detector_destroy(_handle);
        }
    }
}
