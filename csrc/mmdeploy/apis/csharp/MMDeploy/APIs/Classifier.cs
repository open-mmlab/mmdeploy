using System.Collections.Generic;

namespace MMDeploy
{
    /// <summary>
    /// Single classification result of a picture.
    /// A picture may contains multiple reuslts.
    /// </summary>
    public struct Label
    {
        /// <summary>
        /// Id.
        /// </summary>
        public int Id;

        /// <summary>
        /// Score.
        /// </summary>
        public float Score;

        /// <summary>
        /// Initializes a new instance of the <see cref="Label"/> struct.
        /// </summary>
        /// <param name="id">id.</param>
        /// <param name="score">score.</param>
        public Label(int id, float score)
        {
            Id = id;
            Score = score;
        }
    }

    /// <summary>
    /// Output of Classifier.
    /// </summary>
    public struct ClassifierOutput
    {
        /// <summary>
        /// Classification results for single image.
        /// </summary>
        public List<Label> Results;

        /// <summary>
        /// Add result to single image.
        /// </summary>
        /// <param name="id">id.</param>
        /// <param name="score">score.</param>
        public void Add(int id, float score)
        {
            if (Results == null)
            {
                Results = new List<Label>();
            }

            Results.Add(new Label(id, score));
        }

        /// <summary>
        /// Gets number of output.
        /// </summary>
        public int Count
        {
            get { return (Results == null) ? 0 : Results.Count; }
        }
    }

    /// <summary>
    /// classifier.
    /// </summary>
    public class Classifier : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Classifier"/> class.
        /// </summary>
        /// <param name="modelPath">model path.</param>
        /// <param name="deviceName">device name.</param>
        /// <param name="deviceId">deviceId.</param>
        public Classifier(string modelPath, string deviceName, int deviceId)
        {
            ThrowException(NativeMethods.mmdeploy_classifier_create_by_path(modelPath, deviceName, deviceId, out _handle));
        }

        /// <summary>
        /// Get label information of each image in a batch.
        /// </summary>
        /// <param name="mats">input mats.</param>
        /// <returns>Results of each input mat.</returns>
        public List<ClassifierOutput> Apply(Mat[] mats)
        {
            List<ClassifierOutput> output = new List<ClassifierOutput>();
            unsafe
            {
                Label* results = null;
                int* resultCount = null;
                fixed (Mat* _mats = mats)
                {
                    ThrowException(NativeMethods.mmdeploy_classifier_apply(_handle, _mats, mats.Length, &results, &resultCount));
                }

                FormatResult(mats.Length, resultCount, results, ref output, out var total);
                ReleaseResult(results, resultCount, total);
            }

            return output;
        }

        private unsafe void FormatResult(int matCount, int* resultCount, Label* results, ref List<ClassifierOutput> output, out int total)
        {
            total = matCount;
            for (int i = 0; i < matCount; i++)
            {
                ClassifierOutput outi = default;
                for (int j = 0; j < resultCount[i]; j++)
                {
                    outi.Add(results->Id, results->Score);
                    results++;
                }

                output.Add(outi);
            }
        }

        private unsafe void ReleaseResult(Label* results, int* resultCount, int count)
        {
            NativeMethods.mmdeploy_classifier_release_result(results, resultCount, count);
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_classifier_destroy(_handle);
        }
    }
}
