import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

public class Utils {
    public static Mat loadImage(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        return bufferedImage2Mat(img);
    }
    public static Mat bufferedImage2Mat(BufferedImage img) {
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        return new Mat(img.getHeight(), img.getWidth(), img.getColorModel().getNumComponents(),
                PixelFormat.BGR, DataType.INT8, data);
    }
}
