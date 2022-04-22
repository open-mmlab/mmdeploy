using System;
using System.Collections.Generic;
using System.Linq;

namespace MMDeploySharp
{
    unsafe public struct mm_text_recognize_t
    {
        public char* text;
        public float* score;
        public int length;
    }

    public struct TextRecognizerOutput
    {
        public struct BoxOutput
        {
            public unsafe BoxOutput(mm_text_recognize_t* result)
            {
                text = new byte[result->length];
                score = new float[result->length];
                fixed (byte* _text = text)
                {
                    int nbytes = result->length;
                    Buffer.MemoryCopy(result->text, _text, nbytes, nbytes);
                }
                fixed (float* _score = score)
                {
                    int nbytes = result->length * sizeof(float);
                    Buffer.MemoryCopy(result->score, _score, nbytes, nbytes);
                }
            }
            public byte[] text;
            public float[] score;
        }

        public void Init()
        {
            if (boxes == null)
            {
                boxes = new List<BoxOutput>();
            }
        }

        public unsafe void Add(mm_text_recognize_t* result)
        {
            Init();
            boxes.Add(new BoxOutput(result));
        }

        public List<BoxOutput> boxes;
    }

    public class TextRecognizer : DisposableObject
    {

        public TextRecognizer(String model_path, String device_name, int device_id)
        {
            ThrowException(NativeMethods.c_mmdeploy_text_recognizer_create_by_path(model_path, device_name, device_id, out _handle));
        }

        public List<TextRecognizerOutput> Apply(mm_mat_t[] mats)
        {
            List<TextRecognizerOutput> output = new List<TextRecognizerOutput>();
            unsafe
            {
                mm_text_recognize_t* results = null;
                fixed (mm_mat_t* _mats = mats)
                {
                    ThrowException(NativeMethods.c_mmdeploy_text_recognizer_apply(_handle, _mats, mats.Length, &results));
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

        public List<TextRecognizerOutput> Apply(mm_mat_t[] mats, List<TextDetectorOutput> vdetects)
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
                mm_text_detect_t[] bboxes = new mm_text_detect_t[sz];
                int pos = 0;
                for (int i = 0; i < vdetects.Count; i++)
                {
                    for (int j = 0; j < vdetects[i].Count; j++)
                    {
                        bboxes[pos++] = vdetects[i].detects[j];
                    }
                }

                mm_text_recognize_t* results = null;
                fixed (mm_mat_t* _mats = mats)
                fixed (mm_text_detect_t* _bboxes = bboxes)
                fixed (int* _bbox_count = bbox_count)
                {
                    ThrowException(NativeMethods.c_mmdeploy_text_recognizer_apply_bbox(_handle, _mats, mats.Length, _bboxes, _bbox_count, &results));
                    FormatResult(mats.Length, _bbox_count, results, ref output, out var total);
                    ReleaseResult(results, total);
                }
            }

            return output;
        }

        public unsafe void FormatResult(int mat_count, int* result_count, mm_text_recognize_t* results, ref List<TextRecognizerOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < mat_count; i++)
            {
                TextRecognizerOutput outi = new TextRecognizerOutput();
                for (int j = 0; j < result_count[i]; j++)
                {
                    outi.Add(results);
                    results++;
                    total++;
                }
                output.Add(outi);
            }
        }

        public unsafe void ReleaseResult(mm_text_recognize_t* results, int count)
        {
            NativeMethods.c_mmdeploy_text_recognizer_release_result(results, count);
        }

        protected override void ReleaseHandle()
        {
            NativeMethods.c_mmdeploy_text_recognizer_destroy(_handle);
        }
    }
}
