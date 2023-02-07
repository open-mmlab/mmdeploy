namespace MMDeploy
{
    /// <summary>
    /// Context.
    /// </summary>
    public class Context : DisposableObject
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Context"/> class.
        /// </summary>
        public Context()
        {
            ThrowException(NativeMethods.mmdeploy_context_create(out _handle));
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Context"/> class with device.
        /// </summary>
        /// <param name="device">device.</param>
        public Context(Device device) : this()
        {
            Add(device);
        }

        /// <summary>
        /// Add model to the context.
        /// </summary>
        /// <param name="name">name.</param>
        /// <param name="model">model.</param>
        public void Add(string name, Model model)
        {
            ThrowException(NativeMethods.mmdeploy_context_add(this, (int)ContextType.MODEL, name, model));
        }

        /// <summary>
        /// Add scheduler to the context.
        /// </summary>
        /// <param name="name">name.</param>
        /// <param name="scheduler">scheduler.</param>
        public void Add(string name, Scheduler scheduler)
        {
            ThrowException(NativeMethods.mmdeploy_context_add(this, (int)ContextType.SCHEDULER, name, scheduler));
        }

        /// <summary>
        /// Add device to the context.
        /// </summary>
        /// <param name="device">device.</param>
        public void Add(Device device)
        {
            ThrowException(NativeMethods.mmdeploy_context_add(this, (int)ContextType.DEVICE, "", device));
        }

        /// <summary>
        /// Add profiler to the context.
        /// </summary>
        /// <param name="profiler">profiler.</param>
        public void Add(Profiler profiler)
        {
            ThrowException(NativeMethods.mmdeploy_context_add(this, (int)ContextType.PROFILER, "", profiler));
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_model_destroy(_handle);
        }
    }
}
