package mmdeploy;

public class Profiler {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    public final long profiler_;
    private String path_;

    public Profiler(String path) {
        path_ = path;
        profiler_ = create(path);
    }

    public void release() {
        destroy(profiler_);
    }

    private native long create(String path);

    private native void destroy(long profiler_);
}
