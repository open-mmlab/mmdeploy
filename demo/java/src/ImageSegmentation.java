import mmdeploy.Segmentor;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

/** @description: this is a class for ImageSegmentation java demo. */
public class ImageSegmentation {

    /** The main function for ImageSegmentation Java demo.
     * @param deviceName: the device name of the demo.
     * @param modelPath: the image segmentation model path.
     * @param imagePath: the image path.
     */
    public static void main(String[] args) {
        // Parse arguments
        if (args.length != 3) {
            System.out.println("usage:\njava ImageSegmentation deviceName modelPath imagePath");
            return;
        }
        String deviceName = args[0];
        String modelPath = args[1];
        String imagePath = args[2];

        // create segmentor
        Segmentor segmentor = null;

        try {
            segmentor = new Segmentor(modelPath, deviceName, 0);

            // load image
            Mat img = Utils.loadImage(imagePath);

            // apply segmentor
            Segmentor.Result[] result = segmentor.apply(img);

            // print results
            for (Segmentor.Result value : result) {
                System.out.printf("mask height=%d, width=%d\n", value.height, value.width);
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release segmentor
            if (segmentor != null) {
                segmentor.release();
            }
        }
    }
}
