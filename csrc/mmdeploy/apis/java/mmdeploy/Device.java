package mmdeploy;

/** @description: the Device class. */
public class Device {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long deviceHandle;
    private String deviceName;
    private int deviceIndex;

    /** Initialize a new instance of the Device class.
     * @param name: device name.
     * @param index: device index.
    */
    public Device(String name, int index) {
        deviceName = name;
        deviceIndex = index;
        deviceHandle = create(deviceName, deviceIndex);
    }

    /** Get device name.
     * @return: device name.
    */
    public String name() {
        return deviceName;
    }

    /** Get device index.
     * @return: device index.
    */
    public int index() {
        return deviceIndex;
    }

    /** Get device handle.
     * @return: device handle.
    */
    public long handle() {
        return deviceHandle;
    }

    /** Release the instance of Device. */
    public void release() {
        destroy(deviceHandle);
    }

    private native long create(String name, int index);

    private native void destroy(long deviceHandle);
}
