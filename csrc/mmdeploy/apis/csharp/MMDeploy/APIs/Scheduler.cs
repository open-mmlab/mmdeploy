using System;

namespace MMDeploy
{
    /// <summary>
    /// Scheduler.
    /// </summary>
    public class Scheduler : DisposableObject
    {
        private Scheduler()
        {
        }

        /// <summary>
        /// Create thread pool scheduler.
        /// </summary>
        /// <param name="num_threads">thread number.</param>
        /// <returns>scheduler.</returns>
        public static Scheduler ThreadPool(int num_threads)
        {
            Scheduler result = new Scheduler();
            unsafe
            {
                result._handle = (IntPtr)NativeMethods.mmdeploy_executor_create_thread_pool(num_threads);
            }

            return result;
        }

        /// <summary>
        /// Create single thread scheduler.
        /// </summary>
        /// <returns>scheduler.</returns>
        public static Scheduler Thread()
        {
            Scheduler result = new Scheduler();
            unsafe
            {
                result._handle = (IntPtr)NativeMethods.mmdeploy_executor_create_thread();
            }

            return result;
        }

        /// <inheritdoc/>
        protected override void ReleaseHandle()
        {
            NativeMethods.mmdeploy_scheduler_destroy(_handle);
        }
    }
}
