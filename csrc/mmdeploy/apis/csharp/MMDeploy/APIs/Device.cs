namespace MMDeploy
{
    /// <summary>
    /// Device.
    /// </summary>
    public class Device : DisposableObject
    {
        private readonly string _name;
        private readonly int _index;

        /// <summary>
        /// Initializes a new instance of the <see cref="Device"/> class.
        /// </summary>
        /// <param name="name">device name.</param>
        /// <param name="index">device index.</param>
        public Device(string name, int index = 0)
        {
            this._name = name;
            this._index = index;
            ThrowException(NativeMethods.mmdeploy_device_create(name, index, out _handle));
        }

        /// <summary>
        /// Gets device name.
        /// </summary>
        public string Name { get => _name; }

        /// <summary>
        /// Gets device index.
        /// </summary>
        public int Index { get => _index; }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_device_destroy(_handle);
        }
    }
}
