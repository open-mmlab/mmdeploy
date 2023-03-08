package mmdeploy;

/** @description: Mat. */
public class Mat {

    /** Shape. */
    public int[] shape;

    /** Pixel format. */
    public int format;

    /** Data type. */
    public int type;

    /** Mat data. */
    public byte[] data;

    /** Initialize a new instance of the Mat class.
     * @param height: height.
     * @param width: width.
     * @param channel: channel.
     * @param format: pixel format.
     * @param type: data type.
     * @param data: mat data.
    */
    public Mat(int height, int width, int channel,
               PixelFormat format, DataType type, byte[] data) {
        shape = new int[]{height, width, channel};
        this.format = format.value;
        this.type = type.value;
        this.data = data;
    }
}
