namespace MMDeploy
{
    /// <summary>
    /// model.
    /// </summary>
    public class Model : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Model"/> class.
        /// </summary>
        /// <param name="modelPath">model path.</param>
        public Model(string modelPath)
        {
            ThrowException(NativeMethods.mmdeploy_model_create_by_path(modelPath, out _handle));
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_model_destroy(_handle);
        }
    }
}
