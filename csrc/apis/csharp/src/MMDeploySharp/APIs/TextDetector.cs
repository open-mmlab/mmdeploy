using System;
using System.Collections.Generic;

namespace MMDeploySharp
{
    public struct TextBox
    {
        public mm_pointf_t p1;
        public mm_pointf_t p2;
        public mm_pointf_t p3;
        public mm_pointf_t p4;
        public mm_pointf_t this[int i]
        {
            readonly get
            {
                return i switch
                {
                    0 => p1,
                    1 => p2,
                    2 => p3,
                    3 => p4,
                    _ => throw new ArgumentOutOfRangeException(nameof(i))
                };
            }
            set
            {
                switch (i)
                {
                    case 0: p1 = value; break;
                    case 1: p2 = value; break;
                    case 2: p3 = value; break;
                    case 3: p4 = value; break;
                    default: throw new ArgumentOutOfRangeException(nameof(i));
                }
            }
        }
    }

    public struct mm_text_detect_t
    {
        public mm_text_detect_t(float score, TextBox bbox)
        {
            this.score = score;
            this.bbox = bbox;
        }

        public unsafe mm_text_detect_t(mm_text_detect_t* result)
        {
            this.score = result->score;
            bbox = new TextBox();
            for (int i = 0; i < 4; i++)
            {
                this.bbox[i] = result->bbox[i];
            }
        }
        public TextBox bbox;
        public float score;
    }

    public struct TextDetectorOutput
    {
        void Init()
        {
            if (detects == null)
            {
                detects = new List<mm_text_detect_t>();
            }
        }

        public void Add(float score, TextBox bbox)
        {
            Init();
            detects.Add(new mm_text_detect_t(score, bbox));
        }

        public unsafe void Add(mm_text_detect_t* result)
        {
            Init();
            detects.Add(new mm_text_detect_t(result));
        }

        public int Count
        {
            get { return (detects == null) ? 0 : detects.Count; }
        }

        public List<mm_text_detect_t> detects;
    }

    public class TextDetector : DisposableObject
    {
        public TextDetector(String model_path, String device_name, int device_id)
        {
            ThrowException(NativeMethods.c_mmdeploy_text_detector_create_by_path(model_path, device_name, device_id, out _handle));
        }

        public List<TextDetectorOutput> Apply(mm_mat_t[] mats)
        {
            List<TextDetectorOutput> output = new List<TextDetectorOutput>();
            unsafe
            {
                mm_text_detect_t* results = null;
                int* result_count = null;
                fixed (mm_mat_t* _mats = mats)
                {
                    ThrowException(NativeMethods.c_mmdeploy_text_detector_apply(_handle, _mats, mats.Length, &results, &result_count));
                }

                FormatResult(mats.Length, result_count, results, ref output, out var total);
                ReleaseResult(results, result_count, total);
            }

            return output;
        }

        public unsafe void FormatResult(int mat_count, int* result_count, mm_text_detect_t* results, ref List<TextDetectorOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < mat_count; i++)
            {
                TextDetectorOutput outi = new TextDetectorOutput();
                for (int j = 0; j < result_count[i]; j++)
                {
                    outi.Add(results);
                    results++;
                    total++;
                }
                output.Add(outi);
            }
        }

        public unsafe void ReleaseResult(mm_text_detect_t* results, int* result_count, int count)
        {
            NativeMethods.c_mmdeploy_text_detector_release_result(results, result_count, count);
        }

        protected override void ReleaseHandle()
        {
            NativeMethods.c_mmdeploy_text_detector_destroy(_handle);
        }
    }
}
