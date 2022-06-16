package mmdeploy;

public enum DataType {
    FLOAT(0),
    HALF(1),
    INT8(2),
    INT32(3);
    final int value;

    DataType(int value) {
        this.value = value;
    }
}
