package mmdeploy;

public class Profiler {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long profilerHandle;

    public Profiler(String path) {
        profilerHandle = create(path);
    }

    public void release() {
        destroy(profilerHandle);
    }

    public long handle() {
        return profilerHandle;
    }

    private native long create(String path);

    private native void destroy(long profilerHandle);
}
