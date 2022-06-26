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
            for (int i = 0; i < result.length; i++) {
                Detector.Result value = result[i];
                System.out.printf("box %d, left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, label=%d, score=%.4f\n",
                                  i, value.bbox.left, value.bbox.top, value.bbox.right, value.bbox.bottom, value.label_id, value.score);
                if ((value.bbox.right - value.bbox.left) < 1 || (value.bbox.bottom - value.bbox.top) < 1) {
                    continue;
                }

                // skip detections less than specified score threshold
                if (value.score < 0.3) {
                    continue;
                }
                if (value.mask != null) {
                    System.out.printf("mask %d, height=%d, width=%d\n", i, value.mask.shape[0], value.mask.shape[1]);
                }
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
