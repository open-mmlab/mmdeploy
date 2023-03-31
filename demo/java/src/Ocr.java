import mmdeploy.TextDetector;
import mmdeploy.TextRecognizer;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

/** @description: this is a class for Ocr java demo. */
public class Ocr {

    /** The main function for Ocr Java demo.
     * @param deviceName: the device name of the demo.
     * @param detModelPath: the text detection model path.
     * @param recModelPath: the text recognition model path.
     * @param imagePath: the image path.
     */
    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 4) {
            System.out.println("usage:\njava Ocr deviceName detModelPath recModelPath imagePath");
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
            Mat img = Utils.loadImage(imagePath);

            // apply text detector
            TextDetector.Result[] detResult = text_detector.apply(img);

            int [] detResultCount = {detResult.length};
            TextRecognizer.Result[] recResult = text_recognizer.applyBbox(img, detResult, detResultCount);
            if (recResult == null) {
                System.out.println("Apply TextRecognizer failed.");
                System.exit(1);
            }
            // print results
            for (int i = 0; i < detResultCount[0]; ++i) {
                System.out.printf("box[%d]: %s\n", i, new String(recResult[i].text));
                for (int j = 0; j < 4; ++j) {
                    System.out.printf("x: %.2f, y: %.2f, ", detResult[i].bbox[j].x, detResult[i].bbox[j].y);
                }
                System.out.printf("\n");
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release text detector and recognizer
            if (text_recognizer != null) {
                text_recognizer.release();
            }
            if (text_detector != null) {
                text_detector.release();
            }
        }
    }
}
