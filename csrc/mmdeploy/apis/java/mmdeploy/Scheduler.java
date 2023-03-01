package mmdeploy;

/** @description: the Scheduler class. */
public class Scheduler {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private static long schedulerHandle;

    /** Initialize a new instance of the Scheduler class.
     * @param scheduler: scheduler handle.
    */
    public Scheduler(long scheduler) {
        schedulerHandle = scheduler;
    }

    /** Create thread pool scheduler.
     * @param numThreads: thread number.
     * @return: scheduler handle.
    */
    public static long threadPool(int numThreads) {
        schedulerHandle = createThreadPool(numThreads);
        return schedulerHandle;
    }

    /** Create single thread scheduler.
     * @return: scheduler handle.
    */
    public static long thread() {
        schedulerHandle = createThread();
        return schedulerHandle;
    }

    /** Get scheduler handle.
     * @return: scheduler handle.
    */
    public long handle() {
        return schedulerHandle;
    }

    /** Release the instance of Scheduler. */
    public void release() {
        destroy(schedulerHandle);
    }

    private static native long createThreadPool(int numThreads);

    private static native long createThread();

    private native void destroy(long schedulerHandle);
}
