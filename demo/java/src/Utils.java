import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.*; // imread, imwrite, etc

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.lang.*;

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
    public static Mat cvMatToMat(org.opencv.core.Mat cvMat)
    {
        byte[] dataPointer = new byte[cvMat.rows() * cvMat.cols() * cvMat.channels() * (int)cvMat.elemSize()];
        cvMat.get(0, 0, dataPointer);
        Mat mat = new Mat(cvMat.rows(), cvMat.cols(), cvMat.channels(),
                             PixelFormat.BGR, DataType.INT8, dataPointer);
        return mat;
    }
}
