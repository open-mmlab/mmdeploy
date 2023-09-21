import mmdeploy.PoseDetector;
import mmdeploy.PixelFormat;
import mmdeploy.DataType;
import mmdeploy.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Point;
import org.opencv.core.Scalar;

/** @description: this is a class for PoseDetection java demo. */
public class PoseDetection {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

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

            // load image
            org.opencv.core.Mat cvimg = Imgcodecs.imread(imagePath);
            Mat img = Utils.cvMatToMat(cvimg);

            // apply pose estimator
            PoseDetector.Result[] result = poseEstimator.apply(img);

            // print results
            for (PoseDetector.Result value : result) {
                int num_bbox = value.bbox.length;
                int kpt_each_bbox = value.point.length / num_bbox;
                for (int i = 0; i < num_bbox; i++) {
                    System.out.printf("bbox %d\n", i);
                    System.out.printf("left: %f, top: %f, right: %f, bottom: %f\n", value.bbox[i].left,
                        value.bbox[i].top, value.bbox[i].right, value.bbox[i].bottom);

                    int base_idx = kpt_each_bbox * i;
                    for (int j = base_idx; j < base_idx + kpt_each_bbox; j++) {
                        System.out.printf("point %d, x: %d, y: %d\n", i, (int)value.point[j].x, (int)value.point[j].y);
                    }
                    System.out.printf("\n");
                }
            }

            // save image
            for (PoseDetector.Result value : result) {
                for (int i = 0; i < value.bbox.length; i++) {
                    Point pt1 = new Point((int)value.bbox[i].left, (int)value.bbox[i].top);
                    Point pt2 = new Point((int)value.bbox[i].right, (int)value.bbox[i].bottom);
                    Scalar color = new Scalar(0, 0, 255);
                    int thickness = 2;
                    Imgproc.rectangle(cvimg, pt1, pt2, color, thickness);
                }
                for (int i = 0; i < value.point.length; i++) {
                    Point center = new Point((int)value.point[i].x, (int)value.point[i].y);
                    int radius = 2;
                    Scalar color = new Scalar(0, 255, 0);
                    int thickness = 2;
                    Imgproc.circle(cvimg, center, radius, color, thickness);
                }
            }
            Imgcodecs.imwrite("output_pose.jpg", cvimg);
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
