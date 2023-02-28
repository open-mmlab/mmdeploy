package mmdeploy;

public class Context {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long contextHandle;

    public enum ContextType {
        DEVICE,
        SCHEDULER,
        MODEL,
        PROFILER
    }

    public Context() {
        contextHandle = create();
    }

    public void add(String name, Model model) {
        add(contextHandle, ContextType.MODEL.ordinal(), name, model.handle());
    }

    public void add(String name, Scheduler scheduler) {
        add(contextHandle, ContextType.SCHEDULER.ordinal(), name, scheduler.handle());
    }

    public void add(Device device) {
        add(contextHandle, ContextType.DEVICE.ordinal(), "", device.handle());
    }

    public void add(Profiler profiler) {
        add(contextHandle, ContextType.PROFILER.ordinal(), "", profiler.handle());
    }

    public void release() {
        destroy(contextHandle);
    }

    public long handle() {
        return contextHandle;
    }

    private native long create();

    public native int add(long context, int contextType, String name, long handle);

    private native void destroy(long context);
}
