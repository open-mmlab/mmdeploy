package mmdeploy;

public enum PixelFormat {
    BGR(0),
    RGB(1),
    GRAYSCALE(2),
    NV12(3),
    NV21(4),
    BGRA(5);
    final int value;

    PixelFormat(int value) {
        this.value = value;
    }
}
