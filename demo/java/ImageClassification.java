import mmdeploy.Classifier;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

public class ImageClassification {

    private static Mat loadImage(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        return new Mat(img.getHeight(), img.getWidth(), img.getColorModel().getNumComponents(),
                PixelFormat.BGR, DataType.INT8, data);
    }

    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 3) {
            System.out.println("usage:\njava ImageClassification modelPath deviceName imagePath");
            return;
        }
        String modelPath = args[0];
        String deviceName = args[1];
        String imagePath = args[2];

        // create classifier
        Classifier classifier = null;

        try {
            classifier = new Classifier(modelPath, deviceName, 0);
            // load image
            Mat img = loadImage(imagePath);

            // apply classifier
            Classifier.Result[] result = classifier.apply(img);

            // print results
            for (Classifier.Result value : result) {
                System.out.println(value);
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
