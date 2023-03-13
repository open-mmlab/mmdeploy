import mmdeploy.Restorer;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

/** @description: this is a class for ImageRestorer java demo. */
public class ImageRestorer {

    /** The main function for ImageRestorer Java demo.
     * @param deviceName: the device name of the demo.
     * @param modelPath: the image restorer model path.
     * @param imagePath: the image path.
     */
    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 3) {
            System.out.println("usage:\njava ImageRestorer deviceName modelPath imagePath");
            return;
        }
        String deviceName = args[0];
        String modelPath = args[1];
        String imagePath = args[2];

        // create restorer
        Restorer restorer = null;

        try {
            restorer = new Restorer(modelPath, deviceName, 0);

            // load image
            Mat img = Utils.loadImage(imagePath);

            // apply restorer
            Restorer.Result[] result = restorer.apply(img);

            // print results
            for (Restorer.Result value : result) {
                System.out.printf("Restore image height=%d, width=%d\n", value.res.shape[0], value.res.shape[1]);
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release restorer
            if (restorer != null) {
                restorer.release();
            }
        }
    }
}
