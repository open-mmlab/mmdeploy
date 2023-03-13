import mmdeploy.Classifier;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

/** @description: this is a class for ImageClassification java demo. */
public class ImageClassification {

    /** The main function for ImageClassification Java demo.
     * @param deviceName: the device name of the demo.
     * @param modelPath: the image classification model path.
     * @param imagePath: the image path.
     */
    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 3) {
            System.out.println("usage:\njava ImageClassification deviceName modelPath imagePath");
            return;
        }
        String deviceName = args[0];
        String modelPath = args[1];
        String imagePath = args[2];

        // create classifier
        Classifier classifier = null;

        try {
            classifier = new Classifier(modelPath, deviceName, 0);
            // load image
            Mat img = Utils.loadImage(imagePath);

            // apply classifier
            Classifier.Result[] result = classifier.apply(img);

            // print results
            for (Classifier.Result value : result) {
                System.out.printf("label: %d, score: %.4f\n", value.label_id, value.score);
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release classifier
            if (classifier != null) {
                classifier.release();
            }
        }
    }
}
