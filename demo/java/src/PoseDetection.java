import mmdeploy.PoseDetector;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

/**
 * @author: hanrui1sensetime
 * @createDate: 2023/02/28
 * @description: this is a class for PoseDetection java demo.
 */
public class PoseDetection {

    /** The main function for PoseDetection Java demo.
     * @param deviceName: the device name of the demo.
     * @param modelPath: the pose detection model path.
     * @param imagePath: the image path.
     */
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
        PoseDetector poseEstimator = null;

        try {
            poseEstimator = new PoseDetector(modelPath, deviceName, 0);
            if (poseEstimator == -1) {
                System.out.println("Create PoseEstimator failed.");
                System.exit(1);
            }
            // load image
            Mat img = Utils.loadImage(imagePath);

            // apply pose estimator
            PoseDetector.Result[] result = poseEstimator.apply(img);
            if (result == null) {
                System.out.println("Apply PoseEstimator failed.");
                System.exit(1);
            }

            // print results
            for (PoseDetector.Result value : result) {
                for (int i = 0; i < value.point.length; i++) {
                    System.out.printf("point %d, x: %d, y: %d\n", i, (int)value.point[i].x, (int)value.point[i].y);
                }
            }
        } catch (Exception e) {
            System.out.println("exception: " + e.getMessage());
        } finally {
            // release pose estimator
            if (poseEstimator != null) {
                poseEstimator.release();
            }
        }
    }
}
