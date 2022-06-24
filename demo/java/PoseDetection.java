import mmdeploy.PoseDetector;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

public class PoseDetection {

    private static Mat loadImage(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        return new Mat(img.getHeight(), img.getWidth(), img.getColorModel().getNumComponents(),
                PixelFormat.BGR, DataType.INT8, data);
    }

    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 3) {
            System.out.println("usage:\njava PoseDetection deviceName modelPath imagePath");
            return;
        }
        String deviceName = args[0];
        String modelPath = args[1];
        String imagePath = args[2];

        // create pose estimator
        PoseDetector pose_estimator = null;

        try {
            pose_estimator = new PoseDetector(modelPath, deviceName, 0);
            // load image
            Mat img = loadImage(imagePath);

            // apply pose estimator
            PoseDetector.Result[] result = pose_estimator.apply(img);

            // print results
            for (PoseDetector.Result value : result) {
                System.out.println(value);
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release pose estimator
            if (pose_estimator != null) {
                pose_estimator.release();
            }
        }
    }
}
