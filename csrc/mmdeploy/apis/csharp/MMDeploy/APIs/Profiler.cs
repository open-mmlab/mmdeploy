namespace MMDeploy
{
    /// <summary>
    /// Profiler.
    /// </summary>
    public class Profiler : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Profiler"/> class.
        /// </summary>
        /// <param name="path">path.</param>
        public Profiler(string path)
        {
            ThrowException(NativeMethods.mmdeploy_profiler_create(path, out _handle));
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_profiler_destroy(_handle);
        }
    }
}
