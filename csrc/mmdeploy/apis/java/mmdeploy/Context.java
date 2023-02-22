package mmdeploy;

public class Context {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    public final long contextHandle;

    public Context() {
        contextHandle = create();
    }

    public void add(int contextType, String name, long handle) {
        add(contextHandle, contextType, name, handle);
    }

    public void add(int contextType, long handle) {
        add(contextHandle, contextType, "", handle);
    }

    public void release() {
        destroy(contextHandle);
    }

    private native long create();

    public native int add(long context, int contextType, String name, long handle);

    private native void destroy(long context);
}
