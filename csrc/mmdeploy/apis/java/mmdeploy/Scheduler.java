package mmdeploy;

public class Scheduler {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private static long schedulerHandle;

    public Scheduler(long scheduler) {
        schedulerHandle = scheduler;
    }

    public static long threadPool(int numThreads) {
        schedulerHandle = createThreadPool(numThreads);
        return schedulerHandle;
    }

    public static long thread() {
        schedulerHandle = createThread();
        return schedulerHandle;
    }

    public long handle() {
        return schedulerHandle;
    }

    public void release() {
        destroy(schedulerHandle);
    }

    private static native long createThreadPool(int numThreads);

    private static native long createThread();

    private native void destroy(long schedulerHandle);
}
