package mmdeploy;

public class Scheduler {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private static long scheduler_;

    public Scheduler(long scheduler) {
        scheduler_ = scheduler;
    }

    public static long threadPool(int numThreads) {
        scheduler_ = createThreadPool(numThreads);
        return scheduler_;
    }

    public static long thread() {
        scheduler_ = createThread();
        return scheduler_;
    }

    public void release() {
        destroy(scheduler_);
    }

    private static native long createThreadPool(int numThreads);

    private static native long createThread();

    private native void destroy(long scheduler_);
}
