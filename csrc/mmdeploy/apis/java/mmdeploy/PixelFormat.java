package mmdeploy;

/** @description: PixelFormat. */
public enum PixelFormat {
    BGR(0),
    RGB(1),
    GRAYSCALE(2),
    NV12(3),
    NV21(4),
    BGRA(5);
    final int value;

    /** Initialize a new instance of the PixelFormat class.
     * @param value: the value.
    */
    PixelFormat(int value) {
        this.value = value;
    }
}
