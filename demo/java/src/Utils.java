import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import org.opencv.core.*;
import org.opencv.imgcodecs.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.lang.*;

/** @description: this is a util class for java demo. */
public class Utils {

    /** This function loads the image by path.
     * @param path: the image path.
     * @return: the image with Mat format.
     * @exception IOException: throws an IO exception when load failed.
     */
    public static Mat loadImage(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        return bufferedImage2Mat(img);
    }

    /** This function changes bufferedImage to Mat.
     * @param img: the bufferedImage.
     * @return: the image with Mat format.
     */
    public static Mat bufferedImage2Mat(BufferedImage img) {
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        return new Mat(img.getHeight(), img.getWidth(), img.getColorModel().getNumComponents(),
                PixelFormat.BGR, DataType.INT8, data);
    }

    /** This function changes cvMat to Mat.
     * @param cvMat: the image with opencv Mat format.
     * @return: the image with Mat format.
     */
    public static Mat cvMatToMat(org.opencv.core.Mat cvMat)
    {
        byte[] dataPointer = new byte[cvMat.rows() * cvMat.cols() * cvMat.channels() * (int)cvMat.elemSize()];
        cvMat.get(0, 0, dataPointer);
        return new Mat(cvMat.rows(), cvMat.cols(), cvMat.channels(),
                PixelFormat.BGR, DataType.INT8, dataPointer);
    }
}
