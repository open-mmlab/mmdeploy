using System;
using System.Collections.Generic;

namespace MMDeploySharp
{
    public struct mm_class_t
    {
        public int label_id;
        public float score;
    }

    public struct ClassifierOutput
    {
        public struct Label
        {
            public int label_id;
            public float score;

            public Label(int label_id, float score)
            {
                this.label_id = label_id;
                this.score = score;
            }
        }

        public void Add(int label_id, float score)
        {
            if (labels == null)
            {
                labels = new List<Label>();
            }
            labels.Add(new Label(label_id, score));
        }

        public int Count
        {
            get { return (labels == null) ? 0 : labels.Count; }
        }

        public List<Label> labels;
    }

    public class Classifier : DisposableObject
    {
        public Classifier(String model_path, String device_name, int device_id)
        {
            ThrowException(NativeMethods.c_mmdeploy_classifier_create_by_path(model_path, device_name, device_id, out _handle));
        }

        public List<ClassifierOutput> Apply(mm_mat_t[] mats)
        {
            List<ClassifierOutput> output = new List<ClassifierOutput>();
            unsafe
            {
                mm_class_t* results = null;
                int* result_count = null;
                fixed (mm_mat_t* _mats = mats)
                {
                    ThrowException(NativeMethods.c_mmdeploy_classifier_apply(_handle, _mats, mats.Length, &results, &result_count));
                }

                FormatResult(mats.Length, result_count, results, ref output, out var total);
                ReleaseResult(results, result_count, total);
            }

            return output;
        }


        public unsafe void FormatResult(int mat_count, int* result_count, mm_class_t* results, ref List<ClassifierOutput> output, out int total)
        {
            total = 0;
            for (int i = 0; i < mat_count; i++)
            {
                ClassifierOutput outi = new ClassifierOutput();
                for (int j = 0; j < result_count[i]; j++)
                {
                    outi.Add(results[j].label_id, results[j].score);
                    results++;
                    total++;
                }
                output.Add(outi);
            }
        }

        public unsafe void ReleaseResult(mm_class_t* results, int* result_count, int count)
        {
            NativeMethods.c_mmdeploy_classifier_release_result(results, result_count, count);
        }

        protected override void ReleaseHandle()
        {
            NativeMethods.c_mmdeploy_classifier_destroy(_handle);
        }
    }
}