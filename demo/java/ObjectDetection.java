import mmdeploy.Detector;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;
import mmdeploy.Utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

public class ObjectDetection {

    private static Mat loadImage(String path) {
        try {
            return Utils.loadImage(path);
        }
        catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        }
        return null;
    }

    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 3) {
            System.out.println("usage:\njava ObjectDetection deviceName modelPath imagePath");
            return;
        }
        String deviceName = args[0];
        String modelPath = args[1];
        String imagePath = args[2];

        // create detector
        Detector detector = null;

        try {
            detector = new Detector(modelPath, deviceName, 0);
            // load image
            Mat img = loadImage(imagePath);

            // apply detector
            Detector.Result[] result = detector.apply(img);

            // print results
            for (Detector.Result value : result) {
                System.out.println(value);
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release detector
            if (detector != null) {
                detector.release();
            }
        }
    }
}
