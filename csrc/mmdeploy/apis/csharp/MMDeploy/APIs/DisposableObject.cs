using System;
using System.Runtime.InteropServices;

namespace MMDeploy
{
    /// <summary>
    /// Base class which manages its own memory.
    /// </summary>
    public class DisposableObject : IDisposable
    {
#pragma warning disable SA1401 // Fields should be private
        /// <summary>
        /// Handle pointer.
        /// </summary>
        protected IntPtr _handle;
#pragma warning restore SA1401 // Fields should be private

        private bool _disposed = false;

        /// <summary>
        /// Gets a value indicating whether this instance has been disposed.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Close handle.
        /// </summary>
        public void Close()
        {
            Dispose();
        }

        /// <summary>
        /// Releases the resources.
        /// </summary>
        private void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                // Free any other managed objects here.
                ReleaseManaged();
            }

            // Free any unmanaged objects here.
            ReleaseHandle();

            _handle = IntPtr.Zero;

            _disposed = true;
        }

        /// <summary>
        /// Releases managed resources.
        /// </summary>
        protected virtual void ReleaseManaged()
        {
        }

        /// <summary>
        /// Releases unmanaged resources.
        /// </summary>
        protected virtual void ReleaseHandle()
        {
            Marshal.FreeHGlobal(_handle);
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="DisposableObject"/> class.
        /// </summary>
        ~DisposableObject()
        {
            Dispose(false);
        }

        /// <summary>
        /// Throw exception is result is not zero.
        /// </summary>
        /// <param name="result">function return value.</param>
        protected static void ThrowException(int result)
        {
            if (result != 0)
            {
                throw new Exception(result.ToString());
            }
        }

        /// <summary>
        /// Gets internal handle.
        /// </summary>
        /// <param name="obj">instance.</param>
        public static implicit operator IntPtr(DisposableObject obj) => obj._handle;
    }
}
