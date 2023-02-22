package mmdeploy;

public class Profiler {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    public final long profilerHandle;
    private String path_;

    public Profiler(String path) {
        path_ = path;
        profilerHandle = create(path);
    }

    public void release() {
        destroy(profilerHandle);
    }

    private native long create(String path);

    private native void destroy(long profilerHandle);
}
