import mmdeploy.TextDetector;
import mmdeploy.TextRecognizer;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;
import mmdeploy.Utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

public class Ocr {

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
        if (args.length != 4) {
            System.out.println("usage:\njava TextDetection deviceName detModelPath recModelPath imagePath");
            return;
        }
        String deviceName = args[0];
        String detModelPath = args[1];
        String recModelPath = args[2];
        String imagePath = args[3];

        // create text detector and recognizer
        TextDetector text_detector = null;
        TextRecognizer text_recognizer = null;

        try {
            text_detector = new TextDetector(detModelPath, deviceName, 0);
            text_recognizer = new TextRecognizer(recModelPath, deviceName, 0);
            // load image
            Mat img = loadImage(imagePath);

            // apply text detector
            TextDetector.Result[] detResult = text_detector.apply(img);
            int [] detResultCount = new int[] {detResult.length};
            TextRecognizer.Result[] recResult = text_recognizer.apply_bbox(img, detResult, detResultCount);

            // print results
            for (TextRecognizer.Result value : recResult) {
                System.out.println(value);
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release text detector and recognizer
            if (text_detector != null) {
                text_detector.release();
            }
            if (text_recognizer != null) {
                text_recognizer.release();
            }
        }
    }
}
