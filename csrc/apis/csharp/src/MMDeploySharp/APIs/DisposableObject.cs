using System;
using System.Runtime.InteropServices;

namespace MMDeploySharp
{
    public class DisposableObject : IDisposable
    {
        protected IntPtr _handle;

        bool disposed = false;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Close()
        {
            Dispose();
        }

        private void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                // Free any other managed objects here.
                //
            }

            // Free any unmanaged objects here.
            ReleaseHandle();

            _handle = IntPtr.Zero;

            disposed = true;
        }

        protected virtual void ReleaseHandle()
        {
            Marshal.FreeHGlobal(_handle);
        }

        ~DisposableObject()
        {
            Dispose(false);
        }

        protected static void ThrowException(int result)
        {
            if (result != 0)
            {
                throw new Exception();
            }
        }

    }
}
