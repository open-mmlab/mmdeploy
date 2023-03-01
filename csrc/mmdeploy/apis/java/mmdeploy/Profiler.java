package mmdeploy;

/** @description: the Profiler class. */
public class Profiler {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long profilerHandle;

    /** Initialize a new instance of the Profiler class.
     * @param path: profiler path.
    */
    public Profiler(String path) {
        profilerHandle = create(path);
    }

    /** Release the instance of Profiler. */
    public void release() {
        destroy(profilerHandle);
    }

    /** Get profiler handle.
     * @return: profiler handle.
    */
    public long handle() {
        return profilerHandle;
    }

    private native long create(String path);

    private native void destroy(long profilerHandle);
}
