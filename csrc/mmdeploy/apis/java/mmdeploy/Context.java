
package mmdeploy;

/** @description: the Context class. */
public class Context {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long contextHandle;

    /** @description: ContextType. */
    public enum ContextType {
        DEVICE,
        STREAM,
        MODEL,
        SCHEDULER,
        MAT,
        PROFILER
    }

    /** Initializes a new instance of the Context class. */
    public Context() {
        contextHandle = create();
    }

    /** Add Model to the Context.
     * @param name: name.
     * @param model: model.
    */
    public void add(String name, Model model) {
        add(contextHandle, ContextType.MODEL.ordinal(), name, model.handle());
    }

    /** Add Scheduler to the Context.
     * @param name: name.
     * @param scheduler: scheduler.
    */
    public void add(String name, Scheduler scheduler) {
        add(contextHandle, ContextType.SCHEDULER.ordinal(), name, scheduler.handle());
    }

    /** Add Device to the Context.
     * @param device: device.
    */
    public void add(Device device) {
        add(contextHandle, ContextType.DEVICE.ordinal(), "", device.handle());
    }

    /** Add Profiler to the Context.
     * @param profiler: profiler.
    */
    public void add(Profiler profiler) {
        add(contextHandle, ContextType.PROFILER.ordinal(), "", profiler.handle());
    }

    /** Release the instance of Context. */
    public void release() {
        destroy(contextHandle);
    }

    /** Get the handle of Context
     * @return: the handle of context.
    */
    public long handle() {
        return contextHandle;
    }

    private native long create();

    public native int add(long context, int contextType, String name, long handle);

    private native void destroy(long context);
}
